# VIGER AI Telegram Bot

Python-only experimental interface for VIGER SR and the self-trained TinyLM.

## Start

1. Install Python 3.10+.
2. Create a bot with `@BotFather` using `/newbot` and copy the token.
3. Open the VIGER-SR project in VS Code (or any Python editor).
4. Run `run_bot.py`.
5. Paste the token once. It is stored locally in `telegram/.bot_token` and ignored by git.
6. Open the bot and press `/start`.

The launcher checks Python dependencies and offers to install `requirements.txt`. There is no Visual Studio build step, no `.exe`, no CMake build, and no local model server.

## Use

```text
/photo  -> choose AI 2× or 4× -> send image
/text   -> TinyLM generates a response
/teach Hi => Hello!  -> add a training example
/train  -> train the TinyLM from scratch again
/voice on|off -> enable/disable speech replies
/status -> show model/device state
```

## Architecture

```text
Telegram
   │
   ├── photo ──> Python SR engine ──> Swin2SR ──> PNG ──> Telegram
   │
   └── text ───> TinyLM (GRU, trained from scratch) ──> text
                                        │
                                     corpus
                                        │
                                      train
```

Swin2SR is currently used as the image-quality experiment so real 2×/4× super-resolution works without compiling a native runtime. The language model is separate and is trained from scratch from `data/chat.txt` with learned weights only; it does not use a hard-coded fallback or an external chat server.

For the first TinyLM experiment, keep expectations realistic: it is deliberately tiny and learns simple patterns from a small corpus. Add examples with `/teach`, then run `/train` to watch it improve.
