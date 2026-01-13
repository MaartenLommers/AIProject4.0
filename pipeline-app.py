import argparse
import json
import queue
import re
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel
from rapidfuzz import fuzz


# -----------------------------
# Config
# -----------------------------
TARGET_SR = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"
MIN_AUDIO_SECONDS = 0.30

IC_L_PATTERN = re.compile(r"\b(?:IC|L)\s*\d{3,5}\b", re.IGNORECASE)

BELGIUM_CITIES = [
    "Antwerpen", "Brussel", "Bruxelles", "Gent", "Brugge", "Leuven", "Mechelen",
    "Hasselt", "Genk", "Luik", "Liège", "Charleroi", "Namur", "Kortrijk", "Oostende",
    "Aalst", "Mons", "Tournai", "Roeselare",
]

KEYWORDS: Dict[str, List[str]] = {
    "Location-based": ["richting", "tussen", "wissel", "overweg", "talud", "ballast"],
    "Action-based": ["noodrem", "noodstop", "ik sta stil", "ik heb afgeremd", "hoorn", "beveiligd", "cabine beveiligd"],
    "Request-based": ["spoor vrijmaken", "spoor beveiligen", "spanningsloos", "interventie", "inspectie",
                      "politie", "brandweer", "infrabel", "securail", "112", "mug", "ambulance"],
    "People on road": ["persoon op het spoor", "spoorloper", "langs de ballast", "oversteken", "op de sporen",
                       "niet meer in zicht", "hoorn gegeven", "noodrem", "aanrijding"],
    "Obstruction": ["boom op het spoor", "obstructie", "ligt dwars", "beide sporen geblokkeerd", "bovenleiding",
                    "draad zakt door", "hangt", "vonken", "knetter", "spanningsloos", "pantograaf", "tractie af"],
    "Fire": ["rook", "vlammen", "brand", "talud", "naast het spoor", "droog", "breidt uit",
             "evacuatie", "ramen", "deuren dicht", "brandweer"],
    "Collision with object": ["klap", "impact", "iets geraakt", "aanrijding", "onder de trein", "schade",
                              "abnormaal geluid", "luchtverlies", "remdruk", "inspectie onderstel",
                              "stuk materiaal", "fiets", "metaal"],
    "Aggression on board": ["agressieve reiziger", "bedreiging", "scheldt", "duwt", "alcohol",
                            "weigert af te stappen", "securail", "politie", "deuren dicht", "onveilig"],
    "Vehicle on crossing": ["overweg", "auto op de sporen", "slagbomen neer", "lichten knipperen", "vast",
                            "kan niet weg", "verkeer slalomt", "levensgevaarlijk"],
    "Medical Emergency": ["onwel", "ineengezakt", "hart", "reanimeren", "ehbo", "mug",
                          "ambulance", "112", "perron toegang", "vertrek blokkeren"],
    "Animal on road": ["dieren op het spoor", "schapen", "geiten", "koeien", "omheining kapot",
                       "omheining open", "boer", "stapvoets", "stoppen"],
}

URGENCY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

CATEGORY_URGENCY = {
    "Fire": "CRITICAL",
    "People on road": "CRITICAL",
    "Medical Emergency": "CRITICAL",
    "Collision with object": "CRITICAL",
    "Vehicle on crossing": "CRITICAL",
    "Obstruction": "HIGH",
    "Aggression on board": "HIGH",
    "Animal on road": "MEDIUM",
    "Request-based": "MEDIUM",
    "Action-based": "MEDIUM",
    "Location-based": "LOW",
}

URGENT_TERMS = {
    "112", "mug", "ambulance", "brandweer", "politie", "evacuatie",
    "levensgevaarlijk", "noodstop", "noodrem"
}


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_urgency(current: str, new: str) -> str:
    return new if URGENCY_RANK[new] > URGENCY_RANK[current] else current


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def extract_cities(text: str) -> List[str]:
    t = text.lower()
    out: List[str] = []
    seen = set()
    for c in BELGIUM_CITIES:
        if c.lower() in t and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def match_keywords(text: str, threshold: int = 85) -> List[Tuple[str, str]]:
    t = text.lower()
    hits: List[Tuple[str, str]] = []

    for m in IC_L_PATTERN.finditer(text):
        hits.append(("Location-based", m.group(0)))

    for category, words in KEYWORDS.items():
        for kw in words:
            k = kw.lower()
            if k in t or fuzz.partial_ratio(k, t) >= threshold:
                hits.append((category, kw))

    unique: List[Tuple[str, str]] = []
    seen = set()
    for c, k in hits:
        key = (c, k.lower())
        if key not in seen:
            seen.add(key)
            unique.append((c, k))
    return unique


def assign_speakers(segments, gap_seconds: float = 1.2):
    speaker = 0
    last_end = None
    out = []
    for seg in segments:
        if last_end is not None and (seg.start - last_end) > gap_seconds:
            speaker = 1 - speaker
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "speaker": f"SPEAKER_{speaker}",
            "text": seg.text.strip().replace("\n", " ")
        })
        last_end = seg.end
    return out


# -----------------------------
# Preprocess (pure python): mono + 16kHz + simple normalization
# -----------------------------
def linear_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    dur = x.shape[0] / float(orig_sr)
    n_out = int(round(dur * target_sr))
    if n_out <= 1:
        return x[:1]
    xp = np.linspace(0.0, dur, num=x.shape[0], endpoint=False)
    fp = x.astype(np.float32)
    xnew = np.linspace(0.0, dur, num=n_out, endpoint=False)
    return np.interp(xnew, xp, fp).astype(np.float32)


def preprocess_to_16k_mono(in_path: Path, out_path: Path, normalize: bool = True) -> None:
    audio, sr = sf.read(str(in_path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    mono = linear_resample(mono, sr, TARGET_SR)

    if normalize:
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 0:
            mono = mono / peak * 0.95

    sf.write(str(out_path), mono, TARGET_SR, subtype=SUBTYPE)


# -----------------------------
# JSON builder (fits ERD + useful extras)
# -----------------------------
@dataclass
class ModelMeta:
    engine: str
    whisper_model: str
    device: str
    compute_type: str


def analyze_audio_to_json(
    model: WhisperModel,
    clean_wav: Path,
    caller_id: str,
    call_started_iso: str,
    threshold: int,
    lang: str,
    model_meta: ModelMeta,
) -> dict:
    audio_dur = wav_duration_seconds(clean_wav)
    processed_audio_id = str(uuid.uuid4())

    base = {
        "ProcessedAudio": {
            "Id": processed_audio_id,
            "CallerId": caller_id,
            "CallStarted": call_started_iso,
            "CallDuration": int(round(audio_dur)),
            "PriorityLevel": "LOW",
            "DecodedAudio": str(clean_wav),
            "DetectedLanguage": None,
            "TranscribedCall": "",
            "TranslatedCallEnglish": None,
            "TranslatedCallDutch": None,
            "TranslatedCallFrench": None,
            "CreatedAt": now_iso(),
            "UpdatedAt": now_iso(),
        },
        "Keywords": [],
        "Actions": [],
        "Metrics": {
            "model": {
                "engine": model_meta.engine,
                "whisper_model": model_meta.whisper_model,
                "device": model_meta.device,
                "compute_type": model_meta.compute_type,
            },
            "audio_duration_s": audio_dur,
        },
    }

    if audio_dur < MIN_AUDIO_SECONDS:
        base["Metrics"]["error"] = f"Audio too short (<{MIN_AUDIO_SECONDS}s)"
        return base

    t0 = time.perf_counter()
    segments, info = model.transcribe(str(clean_wav), language=lang, vad_filter=True)
    t1 = time.perf_counter()

    turns = assign_speakers(segments)
    transcript_full = " ".join(t["text"] for t in turns).strip()

    # Cities (extra helper field)
    found_cities: List[str] = []
    for tseg in turns:
        found_cities.extend(extract_cities(tseg["text"]))
    found_cities = list(dict.fromkeys(found_cities))  # dedupe

    # Keywords + urgency
    k0 = time.perf_counter()
    summary_counter = {}
    call_urgency = "LOW"
    keyword_rows = []

    for tseg in turns:
        seg_text = tseg["text"]
        seg_lower = seg_text.lower()
        hits = match_keywords(seg_text, threshold=threshold)
        for category, keyword in hits:
            summary_counter[category] = summary_counter.get(category, 0) + 1

            hit_urg = CATEGORY_URGENCY.get(category, "LOW")
            if any(term in seg_lower for term in URGENT_TERMS):
                hit_urg = max_urgency(hit_urg, "HIGH")
            call_urgency = max_urgency(call_urgency, hit_urg)

            keyword_rows.append({
                "Id": str(uuid.uuid4()),
                "word": keyword,
                "context": seg_text,
                "ProcessedAudioId": processed_audio_id,
                "CreatedAt": now_iso(),
                "UpdatedAt": now_iso(),
                "KeywordTime": int(tseg["start"] * 1000),
                "IsAccepted": None,
                # extra MVP fields
                "Category": category,
                "Urgency": hit_urg,
                "Speaker": tseg["speaker"],
            })

    k1 = time.perf_counter()

    actions = []
    if call_urgency in {"HIGH", "CRITICAL"}:
        actions.append({
            "Id": str(uuid.uuid4()),
            "Action": "Escalate / verify emergency protocol",
            "ProcessedAudioId": processed_audio_id,
            "CreatedAt": now_iso(),
            "UpdatedAt": now_iso(),
            "ActionTime": now_iso(),
        })

    base["ProcessedAudio"]["PriorityLevel"] = call_urgency
    base["ProcessedAudio"]["DetectedLanguage"] = getattr(info, "language", None)
    base["ProcessedAudio"]["TranscribedCall"] = transcript_full
    base["ProcessedAudio"]["Locations"] = found_cities  # extra but helpful

    base["Keywords"] = keyword_rows
    base["Actions"] = actions

    base["Metrics"].update({
        "stt_wall_time_s": (t1 - t0),
        "rtf": (t1 - t0) / audio_dur if audio_dur else None,
        "keyword_wall_time_ms": (k1 - k0) * 1000.0,
        "summary": summary_counter,
        "locations_found": len(found_cities),
        "language_probability": float(getattr(info, "language_probability", 0.0)),
        "turns_count": len(turns),
    })

    return base


# -----------------------------
# Recorder (CLI)
# -----------------------------
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def _callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    _audio_q.put(indata.copy())


def record_to_wav(out_path: Path, samplerate: int, channels: int, device: Optional[int] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(
        str(out_path),
        mode="w",
        samplerate=samplerate,
        channels=channels,
        subtype=SUBTYPE,
    ) as wf:
        stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            callback=_callback,
            device=device,
        )
        stream.start()
        print(f"🎙️ Recording... Press ENTER to stop.\nSaved raw to: {out_path.name}")
        input()
        stream.stop()
        stream.close()

        while not _audio_q.empty():
            wf.write(_audio_q.get_nowait())


def main():
    ap = argparse.ArgumentParser(description="Trackle: Record → preprocess → transcribe → JSON (CLI)")
    ap.add_argument("--model", default="medium", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--lang", default="nl", help="Language code (nl)")
    ap.add_argument("--threshold", type=int, default=85, help="Keyword fuzzy threshold")
    ap.add_argument("--caller-id", default="demo_caller", help="CallerId to store in JSON")
    ap.add_argument("--no-normalize", action="store_true", help="Disable preprocessing normalization")
    ap.add_argument("--input-device", type=int, default=None, help="Optional sounddevice input device index")
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "Call_Examples"
    clean_dir = root_dir / "Clean_Audios"
    out_dir = root_dir / "ProcessedAudio_Outputs"
    raw_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = raw_dir / f"recording_{ts}.wav"
    call_started = now_iso()

    record_to_wav(raw_path, TARGET_SR, CHANNELS, device=args.input_device)

    clean_path = clean_dir / f"{raw_path.stem}_16k_mono.wav"
    print("Preprocessing...")
    preprocess_to_16k_mono(raw_path, clean_path, normalize=(not args.no_normalize))

    print("Loading Whisper model...")
    compute_type = "int8" if args.device == "cpu" else "float16"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    meta = ModelMeta(engine="faster-whisper", whisper_model=args.model, device=args.device, compute_type=compute_type)

    print("Transcribing + keyword spotting...")
    analysis = analyze_audio_to_json(
        model=model,
        clean_wav=clean_path,
        caller_id=args.caller_id,
        call_started_iso=call_started,
        threshold=args.threshold,
        lang=args.lang,
        model_meta=meta,
    )

    out_path = out_dir / f"{raw_path.stem}.json"
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    urgency = analysis.get("ProcessedAudio", {}).get("PriorityLevel", "LOW")
    kw_count = len(analysis.get("Keywords", []))
    print(f" Done: {out_path.name} | Priority={urgency} | Keywords={kw_count}")


if __name__ == "__main__":
    main()