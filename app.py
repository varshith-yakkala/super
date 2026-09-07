import os
import streamlit as st
import json
import uuid
from pypdf import PdfReader
from groq import Groq
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Setup Groq client
# Streamlit secrets or env var
api_key = os.environ.get("GROQ_API_KEY")
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

client = None
if api_key:
    # We use instructor to patch the Groq client to enable easy structured Pydantic outputs
    client = instructor.from_groq(Groq(api_key=api_key))

# ----------------- SCHEMAS -----------------

class Fact(BaseModel):
    id: str = Field(description="Unique ID for this fact (e.g., doc1_fact1)")
    entity: str = Field(description="The entity this fact is about (e.g., 'Company X', 'Person Y')")
    attribute: str = Field(description="The attribute or property of the entity (e.g., 'Revenue', 'Job Title')")
    value: str = Field(description="The value of the attribute (e.g., '$10M', 'CEO')")
    context: Optional[str] = Field(description="Any contextual information (e.g., 'Q3 2023', 'In Europe'). Extremely important to capture time, scope, or conditions.")
    evidence: str = Field(description="Exact quote from the document supporting this fact")
    source_doc: str = Field(description="Name of the source document")
    page_num: int = Field(description="Page number where the fact was found")

class FactList(BaseModel):
    facts: List[Fact]

class FactRelationship(BaseModel):
    fact1_id: str
    fact2_id: str
    relationship: Literal["CORROBORATES", "CONTRADICTS", "EXPLAINED_CONTRADICTION", "UNRELATED"] = Field(
        description="CORROBORATES: They state the same underlying truth. CONTRADICTS: They state conflicting things without apparent explanation. EXPLAINED_CONTRADICTION: They appear to contradict but are actually compatible due to context (e.g., different time periods, different units)."
    )
    reasoning: str = Field(description="Detailed explanation for why this relationship was chosen")

class RelationshipList(BaseModel):
    relationships: List[FactRelationship]

# ----------------- HELPERS -----------------

def extract_text_from_pdf(file) -> List[str]:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

def extract_facts_from_text(text: str, doc_name: str, page_num: int) -> List[Fact]:
    if not client:
        st.error("API Key not set.")
        return []
    
    prompt = f"""
    You are an expert fact extractor. Analyze the following text and extract all meaningful numerical or semantic facts.
    Focus on key business metrics, financial figures, organizational roles, important dates, and definitive statements.
    Ensure that 'evidence' is an exact substring from the text.
    If a fact has context (like a time period, geographic region, or condition), capture it in the 'context' field.
    
    Document Name: {doc_name}
    Page Number: {page_num}
    
    Text:
    {text}
    """
    
        import time
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    response_model=FactList,
                    messages=[
                        {"role": "system", "content": "You are a precise data extraction system."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(65)
                else:
                    raise e
        
        result = []
        for f in response.facts:
            # Overwrite source and page to be safe
            f.source_doc = doc_name
            f.page_num = page_num
            # Ensure unique IDs
            if not f.id or "doc" in f.id:
                f.id = str(uuid.uuid4())[:8]
            result.append(f)
        return result
    except Exception as e:
        st.error(f"Failed to parse facts: {e}")
        return []

def compare_facts(new_facts: List[Fact], existing_facts: List[Fact]) -> List[FactRelationship]:
    if not client:
        return []
    if not new_facts or not existing_facts:
        return []
        
    prompt = f"""
    You are a knowledge graph reconciler. Your job is to compare a list of NEW facts with a list of EXISTING facts.
    Identify if any new fact relates to an existing fact.
    
    Relationships:
    - CORROBORATES: They refer to the same entity and attribute, and the values are consistent.
    - CONTRADICTS: They refer to the same entity and attribute, but the values are mutually exclusive and no context explains it.
    - EXPLAINED_CONTRADICTION: They appear to contradict, but the 'context' or 'evidence' shows they are different (e.g. different time periods, different units, different scopes).
    - UNRELATED: They do not refer to the same entity and attribute. (Do not output UNRELATED relationships, only the first three types).
    
    NEW FACTS:
    {[f.model_dump() for f in new_facts]}
    
    EXISTING FACTS:
    {[f.model_dump() for f in existing_facts]}
    """
    
    try:
        import time
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    response_model=RelationshipList,
                    messages=[
                        {"role": "system", "content": "You are a precise fact reconciliation engine."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(65)
                else:
                    raise e
        
        # Filter out unrelated relations
        return [r for r in response.relationships if r.relationship != "UNRELATED"]
    except Exception as e:
        st.error(f"Failed to parse relationships: {e}")
        return []

# ----------------- UI -----------------

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")

st.title("Fact Knowledge Layer")

# Initialize state
if "facts" not in st.session_state:
    st.session_state.facts = []
if "relationships" not in st.session_state:
    st.session_state.relationships = []

with st.sidebar:
    st.header("Configuration")
    if not api_key:
        api_key_input = st.text_input("Groq API Key", type="password")
        if api_key_input:
            client = instructor.from_groq(Groq(api_key=api_key_input))
            st.success("API Key set!")
    else:
        st.success("API Key loaded from environment/secrets.")
        
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files and client:
        with st.spinner("Processing..."):
            for file in uploaded_files:
                doc_name = file.name
                pages_text = extract_text_from_pdf(file)
                for i, text in enumerate(pages_text):
                    new_facts = extract_facts_from_text(text, doc_name, i + 1)
                    if new_facts:
                        # Compare with existing
                        if st.session_state.facts:
                            rels = compare_facts(new_facts, st.session_state.facts)
                            st.session_state.relationships.extend(rels)
                        
                        st.session_state.facts.extend(new_facts)
        st.success("Processing complete!")
    
    if st.button("Clear Knowledge Base"):
        st.session_state.facts = []
        st.session_state.relationships = []
        st.rerun()

st.header("Knowledge Base")
tab1, tab2 = st.tabs(["Facts", "Relationships"])

with tab1:
    if st.session_state.facts:
        facts_data = [f.model_dump() for f in st.session_state.facts]
        df_facts = pd.DataFrame(facts_data)
        st.dataframe(df_facts, use_container_width=True)
    else:
        st.info("No facts extracted yet.")

with tab2:
    if st.session_state.relationships:
        rels_data = [r.model_dump() for r in st.session_state.relationships]
        df_rels = pd.DataFrame(rels_data)
        
        # Add details for easier reading
        def get_fact_text(fact_id):
            for f in st.session_state.facts:
                if f.id == fact_id:
                    return f"{f.entity} - {f.attribute}: {f.value} ({f.context})"
            return "Unknown"
            
        df_rels['Fact 1'] = df_rels['fact1_id'].apply(get_fact_text)
        df_rels['Fact 2'] = df_rels['fact2_id'].apply(get_fact_text)
        
        # Reorder columns
        df_rels = df_rels[['relationship', 'Fact 1', 'Fact 2', 'reasoning', 'fact1_id', 'fact2_id']]
        
        def color_relationship(val):
            color = 'green' if val == 'CORROBORATES' else 'red' if val == 'CONTRADICTS' else 'orange' if val == 'EXPLAINED_CONTRADICTION' else 'white'
            return f'color: {color}'
            
        st.dataframe(df_rels.style.map(color_relationship, subset=['relationship']), use_container_width=True)
    else:
        st.info("No relationships found yet.")
