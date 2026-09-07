import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

m = "qwen/qwen3.8-27b"
try:
    response = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": "Return a valid JSON object with key 'status' and value 'ok'. Must be strict JSON format."}],
        response_format={"type": "json_object"}
    )
    print(f"{m} WORKS! {response.choices[0].message.content}")
except Exception as e:
    print(f"{m} failed: {e}")
