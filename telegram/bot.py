import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = "0.15"
MAX_SECONDS = 180


def find_engine() -> Path:
    configured = os.environ.get("VIGER_SR_EXE")
    candidates = []
    if configured:
        candidates.append(Path(configured))

    build = ROOT / "build"
    if build.exists():
        candidates.extend([
            build / "viger-sr.exe",
            build / "Release" / "viger-sr.exe",
            build / "Debug" / "viger-sr.exe",
        ])
        candidates.extend(sorted(build.rglob("viger-sr.exe")))

    if os.name != "nt":
        candidates.extend([
            build / "viger-sr",
            ROOT / "viger-sr",
        ])
        if build.exists():
            candidates.extend(sorted(build.rglob("viger-sr")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "viger-sr executable not found. Build the C++ core first, or set VIGER_SR_EXE."
    )


def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✨ Enhance 2×", callback_data="scale:2"),
            InlineKeyboardButton("🔥 Enhance 4×", callback_data="scale:4"),
        ],
        [InlineKeyboardButton("ℹ️ Status", callback_data="status")],
    ])


def get_scale(context: ContextTypes.DEFAULT_TYPE) -> int:
    return int(context.user_data.get("scale", 2))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["scale"] = 2
    await update.message.reply_text(
        "⚡ *VIGER SR Experimental Bot*\n\n"
        "Send me a photo and I'll run the current VIGER-SR native pipeline on it.\n\n"
        "Current modes: 2× and 4×.\n"
        "Use the buttons below to choose the scale.",
        reply_markup=keyboard(),
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📸 Send a photo as an image or as a file.\n"
        "⚙️ /start resets the scale to 2×.\n"
        "🔢 Buttons select 2× or 4×.\n\n"
        "The bot uses the native C++ VIGER-SR executable on the machine where the bot is running."
    )


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data.startswith("scale:"):
        scale = int(data.split(":", 1)[1])
        context.user_data["scale"] = scale
        await query.edit_message_text(
            f"✅ Scale set to *{scale}×*.\nSend a photo when ready.",
            reply_markup=keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "status":
        try:
            engine = find_engine()
            text = f"🟢 Engine found:\n`{engine}`\n\nScale: {get_scale(context)}×"
        except FileNotFoundError as exc:
            text = f"🔴 Engine not found.\n`{exc}`"
        await query.edit_message_text(text, reply_markup=keyboard(), parse_mode="Markdown")


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return

    try:
        engine = find_engine()
    except FileNotFoundError as exc:
        await message.reply_text(f"🔴 VIGER SR engine is not ready yet.\n\n{exc}")
        return

    scale = get_scale(context)
    workdir = Path(tempfile.mkdtemp(prefix="viger_sr_bot_"))

    try:
        if message.photo:
            source = message.photo[-1]
            telegram_file = await source.get_file()
            input_path = workdir / "input.jpg"
        elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            source = message.document
            telegram_file = await source.get_file()
            suffix = Path(message.document.file_name or "input.bin").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pnm"}:
                suffix = ".png"
            input_path = workdir / f"input{suffix}"
        else:
            return

        output_path = workdir / "viger_sr_output.png"

        status = await message.reply_text(f"⚙️ Processing… {scale}×")
        await telegram_file.download_to_drive(custom_path=str(input_path))

        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            str(engine),
            str(input_path),
            str(output_path),
            str(scale),
            DEFAULT_DETAIL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=MAX_SECONDS)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            await status.edit_text("⏱️ Processing timed out.")
            return

        elapsed = time.perf_counter() - started
        if process.returncode != 0 or not output_path.exists():
            details = (stderr or stdout).decode("utf-8", errors="replace").strip()
            await status.edit_text(
                "❌ VIGER SR failed.\n\n" + (details[-1800:] or "Unknown error.")
            )
            return

        await status.delete()
        caption = (
            f"✨ VIGER SR\n"
            f"Scale: {scale}×\n"
            f"Time: {elapsed:.2f}s\n\n"
            f"Experimental native pipeline"
        )
        with output_path.open("rb") as output:
            await message.reply_document(document=output, filename="viger_sr.png", caption=caption)
    except Exception as exc:
        await message.reply_text(f"💥 Bot error: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    token = os.environ.get("VIGER_TELEGRAM_TOKEN")
    if not token:
        raise SystemExit(
            "VIGER_TELEGRAM_TOKEN is not set. Create a bot with @BotFather and set the token."
        )

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(callback))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, process_image))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
