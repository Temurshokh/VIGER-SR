# VIGER SR Telegram Bot

Experimental Telegram interface for the native C++ `viger-sr` engine.

## Windows: easiest start

1. Install Python 3 and CMake.
2. Create a bot with `@BotFather` in Telegram using `/newbot`.
3. Clone/pull VIGER-SR.
4. Double-click `start_bot.bat` from this folder.
5. Paste the BotFather token into the terminal.
6. Open your bot, press `/start`, choose `2x` or `4x`, and send a photo.

`start_bot.bat` creates `.venv`, installs the Telegram package, builds the native engine if it cannot find `viger-sr.exe`, and starts long polling.

## PowerShell

```powershell
.\telegram\start_bot.ps1
```

## Existing engine

Set the executable explicitly:

```powershell
$env:VIGER_SR_EXE="C:\path\to\viger-sr.exe"
$env:VIGER_TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
python telegram\bot.py
```

The token is read from `VIGER_TELEGRAM_TOKEN`; it is not stored by the bot.

## What the bot does

```text
Telegram -> bot.py -> viger-sr.exe -> PNG -> Telegram
```

The bot currently exposes the existing native experimental reconstruction pipeline. It does not claim to use trained photographic SR weights unless the neural ONNX pipeline is explicitly wired in later.
