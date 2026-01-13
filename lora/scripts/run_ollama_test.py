import json
import sys
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "emergency-router"


def ollama_generate(prompt: str) -> str:
    """Call Ollama generate API with streaming disabled."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")


def main():
    # Assumes you have already run:
    #   ollama create emergency-router -f lora/ollama/Modelfile
    # so that the model is available locally.
    # Run a few Flemish test prompts
    tests = [
        "Er is rookontwikkeling in het seinhuis van Antwerpen-Berchem; ETCS-monitoren vallen uit.",
        "Bovenleiding is beschadigd ter hoogte van Mechelen; geen spanning op de lijn.",
        "Wisselstoring op spoor 4 in Gent-Sint-Pieters; treinen stapelen op.",
        "Overweg defect in Leuven; slagbomen blijven dicht en verkeer hoopt op.",
        "Meettrein meldt afwijking in spoorstaaf tussen Diest en Aarschot; snelheidsbeperking nodig.",
    ]

    for i, t in enumerate(tests, 1):
        print("\n=== Test", i, "===")
        print("Input:", t)
        out = ollama_generate(t)
        print("Output:", out)


if __name__ == "__main__":
    main()
