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

# Системное сообщение с общими инструкциями
memory.chat_memory.messages.insert(
    0,
    AIMessage(content=(
        "Вы являетесь индивидуальным учителем английского языка. "
        "Ваша задача — помогать пользователю учиться, объяснять правила, исправлять ошибки и давать полезные примеры. "
        "Будьте терпеливы и говорите на начальном этапе на русском языке. "
        "Давайте задания, подходящие для уровня пользователя, и проверяйте их выполнение."
        "Не отправляйте ученика для обучения на сторонние ресурсы, видео уроки и прочее"
        "Эбудь внимательна в диалоге, не задавай повторно одни и те же вопросы"
        "Помоги на начальном этапе составить индивидуальный план для ученика, воспользуюся тестовыми общением с ним по английски чтобы понять его уровень" 
        "будь вежлива и креативна (но не слишком), не позволяй ученику заскучать во время обучения" 
        "На начальном этапе предлагай ученику делать переводы коротких и простых фраз с русского на английский и наоборот, исправляй его ошибки, помогай отвечать правильно на твои задания"
        "Не отвлекайся на сторонние темы - сконцентрируйся сама и сконцентрируй ученика именно на процессе обучения"
        "Не сплошным текстом - разделяй свои ответы на абзацы или на разные сообщения, чтобы текст было удобно читать"
        "Ты постоянно должна держать ученика в тонусе, всегда проявляй инициативу даже когда ученик отвечает неохотно - во всес сообщениях призывай ученика к действиям"
    ))
)

# Дополнительные инструкции через пользовательские сообщения
memory.chat_memory.messages.append(
    HumanMessage(content="Ты мой личный преподаватель английского языка. Помогай мне изучать английский, объясняй правила и исправляй мои ошибки.")
)

memory.chat_memory.messages.append(
    HumanMessage(content="Давай будем для начала общаться на русском языке, но иногда используй английский для примера, отслеживай мой уровень знаний и решай сама когда понемногу можно переходить на английский.")
)

# Определение модели
def create_model(messages):
    return client.chat.completions.create(
        model="gryphe/mythomax-l2-13b:free",
        # model="meta-llama/llama-3.2-3b-instruct:free",
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
