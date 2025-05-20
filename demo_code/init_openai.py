import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(
    api_key=openai_api_key
)
def query_open_ai(prompt=None, temperature=None, model_name=None, max_tokens=None):
    response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()