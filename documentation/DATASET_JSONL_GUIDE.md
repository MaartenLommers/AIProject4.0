# JSONL Dataset Guide for LoRA SFT

This project trains a LoRA adapter using chat-style data. Each line in your JSONL file must be a valid JSON object with a `messages` array. The trainer converts these `messages` into a single training string using the base model's chat template.

## What goes into `messages` and why
- **`messages`**: A list of chat turns. Each turn has a `role` and `content`.
- **Roles**:
  - `system`: High-level instructions for the assistant. Helps steer style and domain (e.g., Flemish railway incidents, keyword spotting, short structured output).
  - `user`: The incident or situation text. This is the input the model should read.
  - `assistant`: The desired response you want the model to learn. Keep it short, practical, and structured so the model consistently outputs your preferred format.
- **Why this format**: `apply_chat_template` stitches these turns into the exact text format expected by the base model. The trainer then learns to produce the `assistant` part when given the `system` and `user`.

## Minimal example (one line)
```json
{"messages":[
  {"role":"system","content":"Je bent een router voor noodgevallen in de Belgische spoorwegen. Geef kort wie te bellen, met sleutelwoorden."},
  {"role":"user","content":"Bovenleiding is beschadigd ter hoogte van Mechelen; geen spanning op de lijn."},
  {"role":"assistant","content":"INCIDENT_TYPE: Schade bovenleiding; KEYWORDS: bovenleiding, spanning; CONTACTS: Infrabel Elektriciteit, Seininfrastructuur, Dispatching"}
]}
```

- The **system** sets the context: emergency contact routing in Flemish railway operations, keyword spotting, short structured output.
- The **user** provides the real-world incident text (Flemish, jargon like "bovenleiding").
- The **assistant** is the target output your model should learn to generate.

## Recommended structure for `assistant` content
Use a concise structure so outputs are consistent and easy to parse:
- `INCIDENT_TYPE`: Short label, e.g., "Wisselstoring" or "Overweg defect".
- `KEYWORDS`: List of the key terms spotted, e.g., `["wissel", "spoor 4"]`.
- `CONTACTS`: Who to call, in order of priority, e.g., `["Infrabel Wisselonderhoud", "Treinverkeersleiding"]`.

Example:
```text
INCIDENT_TYPE: Overweg defect; KEYWORDS: overweg, slagbomen; CONTACTS: Infrabel Seininfrastructuur, Lokale Politie, Verkeerscentrale
```

## Multiple lines
Your `.jsonl` should have one such object per line. For example:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Common pitfalls
- **No comments in JSONL**: JSON doesn't allow comments; keep explanatory notes in separate `.md` files (like this one).
- **Encoding**: Save files as UTF-8 to preserve Flemish characters.
- **Consistency**: Keep your structured format consistent across examples so the model learns a stable pattern.
- **Validation**: Each line must be valid JSON; use tools or scripts to validate if needed.

## Where the trainer looks for data
- In `lora/scripts/train_lora_explained.py` we set:
  - `TRAIN_PATH = "../data/train.jsonl"`
  - `VALID_PATH = "../data/valid.jsonl"`
- These are relative to the `scripts/` folder and point to `lora/data/`.

## Using mock data
- We included Flemish railway mock datasets in `lora/data/mock_train.jsonl` and `lora/data/mock_valid.jsonl`.
- To train quickly on mock data, change the paths in the training script to:
  - `TRAIN_PATH = "../data/mock_train.jsonl"`
  - `VALID_PATH = "../data/mock_valid.jsonl"`

## After training: Ollama
- The adapter files are saved to `lora/output/adapter/`.
- In your `lora/ollama/Modelfile`, reference the adapter:
  - `ADAPTER ./lora/output/adapter/adapter_model.safetensors`
- Build the model:
  - `ollama pull mistral:instruct`
  - `ollama create emergency-router -f lora/ollama/Modelfile`
