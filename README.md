# VIGER SR

**Small model. Big pixels.**

VIGER SR is a Python-first Telegram AI experiment with two independent local inference systems:

- 🖼️ **Image SR** — pretrained Swin2SR for 2× and 4× super-resolution, executed locally.
- 🧠 **VIGER TinyGPT v5** — a ~5M-parameter language model trained from scratch by this project.

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
python run_bot.py
```

You do **not** need to delete your old 1.3M checkpoint if you backed it up separately. V5 has a new model version and architecture, so it will train a new checkpoint rather than silently loading V4.

## Telegram

```text
Telegram
   │
   ├── 📷 photo ──> Python SR engine ──> Swin2SR ──> enhanced image
   │
   └── 💬 text ───> VIGER TinyGPT v5 ───> generated text
```

Commands:

```text
/start
/status
/help
/voice on
/voice off
/learn on
/learn off
/teach Hi => Hello!
/train
/train 5000
/train 8000
/train new 5000
```

`/teach` adds a supervised example to `data/chat.txt`. `/learn on` stores your normal text messages in the local `data/user_learning.txt` corpus so you can feed the model larger language samples without manually writing `/teach` for every line.

## What connects to the internet?

There are two external connections in the normal experiment:

1. **Telegram API** — required for the bot to receive messages and send replies.
2. **Hugging Face model download** — the first time an image SR model is used, its pretrained Swin2SR weights are downloaded. After that, the weights are reused from the local cache.

There is **no remote text-model API** in VIGER TinyGPT. The text architecture, tokenizer, training loop, and learned weights are local.

## Image super-resolution

The current image experiment uses:

```text
2×  caidas/swin2SR-classical-sr-x2-64
4×  caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr
```

The inference path is:

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

The project implements the preprocessing itself with **PIL + NumPy + PyTorch**. `torchvision` is not required by the SR path.

This is genuine neural super-resolution, but it is **not yet a VIGER-trained SR model**. Swin2SR is currently the working image baseline while the text model is built and the future VIGER SR training pipeline is developed.

## VIGER TinyGPT v5

V5 is the first architectural upgrade from the original ~1.3M-parameter VIGER model.

The old 1.3M model is useful as a historical baseline. V5 does not copy those weights; it is a new larger architecture trained from scratch.

```text
local English corpus
       ↓
self-trained byte-level BPE tokenizer
       ↓
<User>/<Assistant>/<System>/<End>
       ↓
VIGER TinyGPT v5
       ↓
next-token probabilities
       ↓
generated response
```

### English corpus

The trainer automatically loads **every `.txt` file in `data/`**.

Current language sources include:

- `english_curriculum.txt` — core vocabulary, grammar, sentence patterns, and simple reasoning;
- `english_natural_c1.txt` — everyday dialogue, contractions, casual phrases, phrasal verbs, short stories, history, technical explanations, and natural conversation;
- `english_advanced_c1.txt` — C1-oriented discourse, uncertainty, polite disagreement, register changes, idioms, nuanced explanations, narrative prose, and advanced grammar;
- `knowledge_base.txt` — a broad factual knowledge pack covering physics, chemistry, biology, astronomy, Earth science, mathematics, computing, AI, history, geography, and everyday concepts;
- `user_learning.txt` — optional local material collected with `/learn on`.

This is a **C1-oriented language corpus plus a broad factual knowledge pack**, not a claim that a 5M model will reach CEFR C1. The goal is to expose the model to richer language structure and a wider set of factual concepts.

### V5 architecture

Current defaults:

```text
Embedding dimension : 256
Transformer layers   : 6
Attention heads      : 8
Context length       : 256 tokens
BPE merges           : up to 512
Dropout              : 0.10
Approx parameters    : ~5M at the current vocabulary
```

The model ties its input and output token embeddings to keep the architecture compact.

Generation:

- minimum 6 new tokens before `<END>`;
- role tokens are blocked during answer generation;
- mild repetition penalty;
- temperature and top-k sampling.

Training:

- AdamW;
- gradient clipping;
- 10% validation split;
- warm-up learning rate;
- cosine learning-rate decay;
- deterministic seed for reproducible experiments.

The checkpoint stores the architecture metadata, tokenizer merges, model version, corpus hash, parameter count, and training steps. A separate local recovery checkpoint stores optimizer and RNG state every 500 steps, so an interrupted CPU run can resume instead of losing hours of computation.

### 1.3M → 5M → 10M → 20M

Parameter count changes only when the architecture changes.

```text
More training data / more steps
        ↓
better values for the same parameters

Larger dim / more layers / larger architecture
        ↓
more parameters
```

The intended progression is:

```text
VIGER 1.3M  ← saved baseline
     ↓
VIGER ~5M   ← current V5
     ↓
VIGER ~10M
     ↓
VIGER ~20M
```

Each size should be evaluated on the same fixed test prompts so improvements can be compared.

## Voice

`/voice on` asks the local `pyttsx3` engine to turn the generated response into audio. Voice generation happens on the machine running the bot.

## Project structure

```text
VIGER-SR/
├── python/
│   ├── viger_tiny_lm.py       ← TinyGPT v5 + BPE + training
│   └── sr_engine.py           ← local Swin2SR inference
├── telegram_bot/
│   └── bot.py                 ← Telegram interface
├── data/
│   ├── chat.txt               ← project chat examples
│   ├── english_curriculum.txt ← core English
│   ├── english_natural_c1.txt ← natural conversation + stories
│   ├── english_advanced_c1.txt  ← advanced English
├── knowledge_base.txt       ← factual knowledge
└── user_learning.txt        ← local user corpus, ignored by Git
├── run_bot.py                 ← one-command launcher
├── run.py                     ← simple Python entry point
└── requirements.txt           ← Python dependencies
```

## Design rule

**Build the interesting parts for real: learned text generation, learned image reconstruction, local inference, and measurable experiments.**
