import uuid
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")

# Инициализация клиента OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# Инициализация памяти
memory = ConversationBufferMemory(return_messages=True)

# Добавляем инструкцию для общения на русском языке
memory.chat_memory.messages.append(
    HumanMessage(content="Давай говорить на русском языке.")
)

# Определение модели
def create_model(messages):
    return client.chat.completions.create(
        model="gryphe/mythomax-l2-13b:free",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>",
        },
    )

# Функция вызова модели
def call_model(chat_id, user_input):
    # Добавляем пользовательское сообщение в память
    memory.chat_memory.messages.append(HumanMessage(content=user_input))
    
    # Преобразуем сообщения для модели
    model_messages = [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content,
        }
        for msg in memory.chat_memory.messages
    ]
    
    # Запрос к модели
    response = create_model(model_messages)
    
    # Ответ модели
    output_message = AIMessage(content=response.choices[0].message.content)
    
    # Сохраняем контекст
    memory.save_context({"content": user_input}, {"content": output_message.content})
    
    # Возвращаем ответ
    return output_message.content
