import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("NO API KEY")
    exit()

client = genai.Client(api_key=api_key)

class Fact(BaseModel):
    id: str = Field(description="Unique ID for this fact")
    entity: str = Field(description="The entity this fact is about")
    attribute: str = Field(description="The attribute or property")
    value: str = Field(description="The value")
    context: Optional[str] = Field(description="Any contextual information")
    evidence: str = Field(description="Exact quote")

class FactList(BaseModel):
    facts: List[Fact]

prompt = "Acme Corp revenue was $10M in Q1 2023."
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FactList,
            temperature=0.0
        ),
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
