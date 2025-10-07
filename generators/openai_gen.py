import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate(prompt: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.2
    )
    # The new response format
    return response.choices[0].message.content.strip()
