# VIGER SR

**Small model. Big pixels.**

VIGER SR is a Python-first Telegram AI experiment with two independent parts:

- 🖼️ real pretrained neural super-resolution for 2×/4× image enhancement;
- 🧠 a tiny language model trained from scratch for simple text generation.

## Fast start

Requirements: Python 3.10+.

```bash
python run_bot.py
```

The launcher automatically creates `.venv`, installs `requirements.txt`, asks for the Telegram BotFather token once, and starts the bot. No Visual Studio, CMake, C++, `.exe`, or local model server is needed for the Telegram experiment.

The token is kept locally in `.bot_token` and is ignored by Git.

## Telegram

```text
Telegram
   │
   ├── 📷 photo ──> Swin2SR 2×/4× ──> enhanced image
   │
   └── 💬 text ───> VIGER TinyLM ───> generated text
```

The bot lives in `telegram_bot/`. The folder is intentionally **not** named `telegram`, because that would shadow the external `python-telegram-bot` package during local imports.

Commands:

```text
/start
/status
/help
/voice on
/voice off
/teach Hi => Hello!
/train
```

`/teach` appends an example to `data/chat.txt`; `/train` trains the TinyLM again from its current corpus. The bot does not use a hard-coded fallback answer for normal text generation.

## Image super-resolution

The Telegram experiment uses local pretrained Swin2SR models:

```text
2×  caidas/swin2SR-classical-sr-x2-64
4×  caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr
```

The first use downloads the model into the local Hugging Face cache. Later runs reuse it locally. CUDA is used automatically when PyTorch detects a compatible GPU; otherwise the model runs on CPU.

The output is capped at 3840×2160 to avoid unnecessarily huge Telegram files.

## Tiny language model

`python/viger_tiny_lm.py` contains a deliberately small byte-level GRU language model trained from scratch. It has learned weights, not a response dictionary:

```text
text -> UTF-8 bytes -> GRU -> next-byte probabilities -> generated text
```

The initial corpus is intentionally tiny and is only for experimentation. It is not a general-purpose language model.

Train manually:

```bash
python python/viger_tiny_lm.py
```

Or from Telegram:

```text
/teach Hi => Hello!
/teach Bye => See you!
/train
```

## Optional native C++ research path

The repository still contains the original C++/CMake research core under `src/`, `include/`, `tests/`, and the corresponding build files. It is **not required** for the Python-only Telegram bot.

## Project structure

```text
VIGER-SR/
├── python/            TinyLM + SR Python engines
├── telegram_bot/      Telegram interface
├── data/              TinyLM training corpus
├── run.py             simple Python entry point
├── run_bot.py         zero-configuration launcher
├── requirements.txt   Python dependencies
├── src/               optional native research core
├── include/           optional native headers
└── tests/             optional native tests
```

## Design rule

Build the experiment so the interesting parts are real: learned image reconstruction, learned text generation, and measurable local inference.
