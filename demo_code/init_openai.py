import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(
    api_key=openai_api_key
)
def query_gpt4(prompt=None, temperature=None):
    response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100
    )
    return response.choices[0].message.content.strip()