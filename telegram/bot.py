"""VIGER Telegram bot: image super-resolution + a tiny self-trained language model."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

ROOT = Path(__file__).resolve().parents[1]

# Lazy imports keep startup messages cleaner and make it obvious which model is loading.
_lm = None
_sr = None


def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✨ AI 2×", callback_data="scale:2"),
            InlineKeyboardButton("🔥 AI 4×", callback_data="scale:4"),
        ],
        [InlineKeyboardButton("🧠 Tiny AI status", callback_data="status")],
    ])


def get_scale(context: ContextTypes.DEFAULT_TYPE) -> int:
    return int(context.user_data.get("scale", 2))


def get_lm():
    global _lm
    if _lm is None:
        from python.viger_tiny_lm import load_or_train
        _lm = load_or_train()
    return _lm


def get_sr():
    global _sr
    if _sr is None:
        from python.sr_engine import SREngine
        _sr = SREngine()
    return _sr


def generate_text(text: str) -> str:
    from python.viger_tiny_lm import answer
    return answer(get_lm(), text)


def make_voice(text: str, path: Path) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(text, str(path))
    engine.runAndWait()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["scale"] = 2
    await update.message.reply_text(
        "⚡ *VIGER AI lab*\n\n"
        "📸 Send a photo → AI 2×/4× super-resolution.\n"
        "💬 Send text → my tiny language model generates the answer.\n"
        "🔊 /voice on — also send the answer as audio.\n\n"
        "No Visual Studio, no .exe and no local model server. Everything runs from Python on this machine.",
        reply_markup=keyboard(),
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📸 Photo: choose AI 2× or AI 4×, then send the image.\n"
        "💬 Text: I'll answer with the locally trained TinyLM.\n"
        "🔊 /voice on|off: enable/disable generated voice replies.\n"
        "ℹ️ /status: show which local models are loaded."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    device = "unknown"
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
    except Exception:
        pass
    await update.message.reply_text(
        f"🧠 VIGER TinyLM: {'loaded' if _lm is not None else 'not loaded yet'}\n"
        f"🖼️ Swin2SR: {'loaded' if _sr is not None else 'not loaded yet'}\n"
        f"⚙️ Device: {device}\n"
        f"📁 Checkpoint: artifacts/viger_tiny_lm.pt"
    )


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    value = (context.args[0].lower() if context.args else "on")
    context.user_data["voice"] = value in {"on", "1", "true", "yes"}
    state = "ON 🔊" if context.user_data["voice"] else "OFF"
    await update.message.reply_text(f"Voice: {state}")


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("scale:"):
        scale = int(data.split(":", 1)[1])
        context.user_data["scale"] = scale
        await query.edit_message_text(
            f"✅ AI scale: *{scale}×*\nSend a photo.",
            reply_markup=keyboard(),
            parse_mode="Markdown",
        )
    elif data == "status":
        device = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device = "CUDA"
        except Exception:
            pass
        await query.edit_message_text(
            f"🧠 TinyLM: {'loaded' if _lm is not None else 'will train on first message'}\n"
            f"🖼️ SR model: {'loaded' if _sr is not None else 'will load on first photo'}\n"
            f"⚙️ Device: {device}",
            reply_markup=keyboard(),
        )


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    scale = get_scale(context)
    workdir = Path(tempfile.mkdtemp(prefix="viger_sr_"))
    status = await message.reply_text(f"⚙️ AI super-resolution {scale}×…")
    try:
        if message.photo:
            telegram_file = await message.photo[-1].get_file()
            input_path = workdir / "input.jpg"
        elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            telegram_file = await message.document.get_file()
            suffix = Path(message.document.file_name or "input.png").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                suffix = ".png"
            input_path = workdir / f"input{suffix}"
        else:
            return

        await telegram_file.download_to_drive(custom_path=str(input_path))
        image = Image.open(input_path).convert("RGB")
        started = time.perf_counter()
        engine = get_sr()
        output = await asyncio.to_thread(engine.enhance, image, scale)
        output_path = workdir / f"viger_ai_{scale}x.png"
        await asyncio.to_thread(engine.save_4k_or_less, output, output_path)
        elapsed = time.perf_counter() - started

        await status.delete()
        caption = (
            f"✨ VIGER AI SR\n"
            f"Scale: {scale}×\n"
            f"Output: {output.width}×{output.height}\n"
            f"Time: {elapsed:.2f}s\n"
            f"Model: Swin2SR"
        )
        with output_path.open("rb") as file:
            await message.reply_document(document=file, filename=output_path.name, caption=caption)
    except Exception as exc:
        await status.edit_text(
            "❌ AI image processing failed.\n\n"
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def process_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    text = message.text.strip()
    if not text:
        return

    status = await message.reply_text("🧠 Thinking…")
    try:
        started = time.perf_counter()
        response = await asyncio.to_thread(generate_text, text)
        elapsed = time.perf_counter() - started
        await status.edit_text(f"{response}\n\n_({elapsed:.2f}s • local TinyLM)_", parse_mode="Markdown")

        if context.user_data.get("voice", False):
            workdir = Path(tempfile.mkdtemp(prefix="viger_voice_"))
            try:
                voice_path = workdir / "reply.wav"
                await asyncio.to_thread(make_voice, response, voice_path)
                if voice_path.exists():
                    with voice_path.open("rb") as audio:
                        await message.reply_voice(voice=audio)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
    except Exception as exc:
        await status.edit_text(
            "🧠 TinyLM crashed while generating.\n\n"
            f"{type(exc).__name__}: {exc}"
        )


def main() -> None:
    token = os.environ.get("VIGER_TELEGRAM_TOKEN") or input("Telegram bot token: ").strip()
    if not token:
        raise SystemExit("Telegram bot token is required.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("voice", voice_command))
    application.add_handler(CallbackQueryHandler(callback))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, process_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text))
    print("[VIGER] Telegram bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
