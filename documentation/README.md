# Audio Recorder
One-click voice recorder. Starts recording immediately and shows a single Stop button. Saves a timestamped WAV into `Prerecorded_Audio/`.

## Install
Use a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
Start the recorder (records at 16 kHz mono, 16‑bit PCM):

```powershell
.venv\Scripts\python.exe audio_capture.py
```

- Click "Stop and Save" to finish. The file is saved as `Prerecorded_Audio/recording_YYYYMMDD_HHMMSS.wav`.

Notes:
- 16 kHz mono is well-suited for human speech with good quality and compact size.
- If your audio device doesn’t support 16 kHz, the app will show an error; we can adjust the rate if needed.
