# VIGER SR

**Small model. Big pixels.**

VIGER SR is a Python-first Telegram AI experiment with two independent local inference systems:

- 🖼️ **Image SR** — pretrained Swin2SR for 2× and 4× super-resolution, executed locally.
- 🧠 **VIGER TinyGPT v3** — a small language model trained from scratch by this project.

The Telegram experiment does **not** require Visual Studio, CMake, C++, a native `.exe`, or a local LLM server.

## Fast start

Requirements: **Python 3.10+**.

```bash
python run_bot.py
```

The launcher creates `.venv`, installs the Python requirements, asks for the Telegram BotFather token once, checks the AI packages, and starts the bot.

Your token is saved locally in `.bot_token` and ignored by Git.

If you update from an older version, the cleanest upgrade is:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force artifacts\viger_tiny_lm.pt -ErrorAction SilentlyContinue
python run_bot.py
```

You do not normally need to delete anything on later runs. The TinyGPT checkpoint stores both a model version and a SHA-256 fingerprint of the training corpus, so changing the corpus automatically triggers a fresh training run.

## Telegram

```text
Telegram
   │
   ├── 📷 photo ──> Python SR engine ──> Swin2SR ──> enhanced image
   │
   └── 💬 text ───> VIGER TinyGPT ───> generated text
```

Commands:

```text
/start
/status
/help
/voice on
/voice off
/teach Hi => Hello!
/train
/train 3000
```

`/teach` adds a real training example to `data/chat.txt`. `/train` retrains the neural model from the updated corpus. Normal replies are generated from model weights; there is no hard-coded response dictionary.

For better image input quality, send the image as a **Telegram file/document** when possible. A normal Telegram chat photo may be compressed before VIGER receives it.

## What connects to the internet?

There are only two external connections in the normal experiment:

1. **Telegram API** — required for the bot to receive your messages and send replies.
2. **Hugging Face model download** — the first time an image SR model is used, its pretrained Swin2SR weights are downloaded. After that, the weights are reused from the local `.cache/huggingface` cache.

There is **no remote text-model API** in VIGER TinyGPT. Its architecture, tokenizer, training loop, and learned weights are all local to this repository/runtime.

## Image super-resolution

The current image experiment uses two pretrained Swin2SR checkpoints:

```text
2×  caidas/swin2SR-classical-sr-x2-64
4×  caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr
```

The inference path is intentionally simple:

```text
PIL image
   ↓
normalize to 0..1
   ↓
pad to dimensions divisible by 8
   ↓
PyTorch tensor [1, 3, H, W]
   ↓
Swin2SR
   ↓
remove padding
   ↓
PNG output, capped at 3840×2160
```

The project performs this preprocessing itself with **PIL + NumPy + PyTorch**. `torchvision` is not required by the SR path.

This is genuine neural super-resolution, but it is **not yet a VIGER-trained SR model**. The pretrained checkpoint is being used as the working image benchmark while we build the rest of the system. Training our own photographic SR model is a later stage and requires a substantially larger dataset and training budget.

## VIGER TinyGPT v3

The language model is deliberately tiny and educational, but it is a real causal Transformer trained from scratch:

```text
text corpus
   ↓
self-trained byte-level BPE tokenizer
   ↓
<User>/<Assistant>/<System>/<End> special tokens
   ↓
TinyGPT Transformer
   ↓
next-token probabilities
   ↓
generated response
```

### Tokenization

The tokenizer starts with raw bytes, then learns frequent byte-pair merges from the corpus. This gives the model a vocabulary of reusable subword-like pieces while keeping a byte fallback for unseen words.

The special role tokens are learned as dedicated IDs:

```text
<SYSTEM>
<USER>
<ASSISTANT>
<END>
```

This means `User` and `Assistant` are not merely formatting in the input: the model is trained to treat the speaker roles as part of its sequence structure.

### Model

Current defaults:

```text
Embedding dimension : 160
Transformer layers   : 4
Attention heads      : 4
Context length       : 192 tokens
BPE merges           : up to 220
```

The checkpoint stores its architecture metadata, tokenizer merges, model version, and corpus hash. When the code or corpus changes, an old incompatible checkpoint is retrained instead of silently being reused.

### Chat context

The Telegram bot keeps the last few user/assistant exchanges in the current chat session and feeds them back into the model context. This is lightweight conversation context, not long-term memory.

### Teaching the model

Example:

```text
/teach What is User? => User is the person who sends a message.
/teach What is Assistant? => Assistant is the program that responds.
/train 3000
```

The first command changes the actual corpus. The second trains the neural weights on that corpus. No response is inserted into a dictionary.

## Voice

`/voice on` asks the local `pyttsx3` engine to turn the generated response into audio. Voice generation is performed on the machine running the bot.

## Running model training directly

```bash
python python/train_tiny_lm.py --steps 2200
```

The compatibility trainer now delegates to the same TinyGPT implementation used by the bot, so there is only one language-model architecture to maintain.

## Project structure

```text
VIGER-SR/
├── python/
│   ├── viger_tiny_lm.py   ← TinyGPT + BPE + training
│   ├── sr_engine.py       ← local Swin2SR inference
│   └── train_tiny_chat.py ← TinyGPT CLI wrapper
├── telegram_bot/
│   └── bot.py             ← Telegram interface
├── data/
│   └── chat.txt           ← language-model training corpus
├── run_bot.py             ← one-command launcher
├── run.py                 ← simple Python entry point
└── requirements.txt       ← Python dependencies
```

## Design rule

**Build the interesting parts for real: learned text generation, learned image reconstruction, local inference, and measurable experiments.**
