import os
from pypdf import PdfReader
from groq import Groq
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
client = instructor.from_groq(Groq(api_key=api_key))

class Fact(BaseModel):
    id: str = Field(description="Unique ID for this fact")
    entity: str = Field(description="The entity this fact is about")
    attribute: str = Field(description="The attribute or property")
    value: str = Field(description="The value")
    context: Optional[str] = Field(description="Any contextual information")
    evidence: str = Field(description="Exact quote")
    source_doc: str = Field(description="Name of the source document")
    page_num: int = Field(description="Page number")

class FactList(BaseModel):
    facts: List[Fact]

class FactRelationship(BaseModel):
    fact1_id: str
    fact2_id: str
    relationship: Literal["CORROBORATES", "CONTRADICTS", "EXPLAINED_CONTRADICTION", "UNRELATED"]
    reasoning: str

class RelationshipList(BaseModel):
    relationships: List[FactRelationship]

def test_run(run_number):
    print(f"--- RUN {run_number} ---")
    files = [
        "data/Doc1_Q1_Report.pdf",
        "data/Doc2_Q2_Report.pdf",
        "data/Doc3_Audit_Report.pdf",
        "data/Doc4_Press_Release.pdf"
    ]
    
    all_facts = []
    
    for file in files:
        reader = PdfReader(file)
        pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
        for i, text in enumerate(pages_text):
            prompt = f"Doc: {os.path.basename(file)}, Page: {i+1}\nText:\n{text}"
            
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        response_model=FactList,
                        messages=[
                            {"role": "system", "content": "You are a precise data extraction system. Extract a MAXIMUM of 4 most critical facts. Keep all fields VERY short to conserve output tokens."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=500
                    )
                    facts = response.facts
                    for f in facts:
                        f.source_doc = os.path.basename(file)
                        f.page_num = i + 1
                        f.id = f"test_{len(all_facts)}"
                        all_facts.append(f)
                    print(f"Extracted {len(facts)} facts from {os.path.basename(file)}")
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        print(f"Rate limited on {os.path.basename(file)}. Waiting 65s...")
                        time.sleep(65)
                    else:
                        print(f"ERROR extraction on {file}: {e}")
                        return False

    print(f"Total facts: {len(all_facts)}")
    
    if len(all_facts) > 0:
        prompt = f"NEW FACTS: {[f.model_dump() for f in all_facts]}\nEXISTING FACTS: []"
        
        for attempt in range(5):
            try:
                rel_prompt = f"Compare these facts. Keep reasoning VERY concise (1 sentence max). Output MAXIMUM 6 most important relationships:\n{[f.model_dump() for f in all_facts]}"
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    response_model=RelationshipList,
                    messages=[
                        {"role": "system", "content": "You are a precise fact reconciliation engine."},
                        {"role": "user", "content": rel_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=950
                )
                rels = [r for r in response.relationships if r.relationship != "UNRELATED"]
                print(f"Found {len(rels)} relationships.")
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    print("Rate limited on relationships. Waiting 65s...")
                    time.sleep(65)
                else:
                    print(f"ERROR in relationships: {e}")
                    return False

    return True

success = True
for i in range(1, 4):
    if not test_run(i):
        success = False
        break

if success:
    print("ALL 3 RUNS SUCCESSFUL!")
else:
    print("FAILED.")
