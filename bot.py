import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from langgraph_agent import call_model  # Импортируем функцию для работы с агентом
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

print(f"Telegram Token: {os.getenv('TELEGRAM_TOKEN')}")

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я ваш ассистент для изучения английского языка. Напишите любое сообщение, чтобы начать!")

# Основной обработчик сообщений
async def handle_message(update: Update, context):
    user_input = update.message.text
    chat_id = str(update.message.chat_id)  # Уникальный ID чата
    try:
        # Запрос к агенту
        response_text = call_model(chat_id, user_input)
        await update.message.reply_text(response_text)
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

# Настройка приложения Telegram
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Запуск бота
app.run_polling()
