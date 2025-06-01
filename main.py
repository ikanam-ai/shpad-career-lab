import openai
import gradio as gr

client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="unused"
)

def get_response(text):
    response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content":text}],
    temperature=0.3,  # Добавляем параметр для креативности
    max_tokens=2000
    )
    return response.choices[0].message.content


# Создаем интерфейс Gradio
demo = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=2, placeholder="Введите текст..."),
    outputs="text",
    title="Анализ настроения текста",
    description="Определяет, позитивный или негативный оттенок у текста."
)

demo.launch()