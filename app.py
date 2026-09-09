import os, re, time, uuid
import streamlit as st
from pypdf import PdfReader
from groq import Groq
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL             = "qwen/qwen3.6-27b"
MAX_OUTPUT_TOKENS = 1200
MAX_PAGE_CHARS    = 3000

api_key = os.environ.get("GROQ_API_KEY")
try:
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

def make_client(key):
    return instructor.from_groq(Groq(api_key=key), mode=instructor.Mode.MD_JSON)

client = make_client(api_key) if api_key else None

class Fact(BaseModel):
    id: str = Field(description="Unique short ID e.g. f1")
    entity: str = Field(description="The entity this fact is about")
    attribute: str = Field(description="The attribute of the entity")
    value: str = Field(description="The value max 8 words")
    context: Optional[str] = Field(None, description="Time scope conditions max 6 words")
    evidence: str = Field(description="Short exact quote max 12 words")
    source_doc: str = Field(description="Source document name")
    page_num: int = Field(description="Page number")

class FactList(BaseModel):
    facts: List[Fact]

class FactRelationship(BaseModel):
    fact1_id: str
    fact2_id: str
    relationship: Literal["CORROBORATES", "CONTRADICTS", "EXPLAINED_CONTRADICTION"] = Field(
        description="CORROBORATES same truth. CONTRADICTS mutually exclusive. EXPLAINED_CONTRADICTION context explains it."
    )
    reasoning: str = Field(description="One sentence max 15 words")

class RelationshipList(BaseModel):
    relationships: List[FactRelationship]

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text)
    return pages

def calculate_fact_density(text):
    numbers    = len(re.findall(r'\b\d+[\d,.]*\b', text))
    currencies = len(re.findall(r'[\$\xa3\u20ac\u20b9]', text))
    kw = len(re.findall(
        r'(?i)\b(revenue|profit|loss|margin|ebitda|growth|ceo|director|increase|decrease|'
        r'sales|capital|debt|equity|asset|liability|percent|gdp|inflation|rate|forecast|quarter|annual)\b', text))
    return numbers + (currencies * 2) + (kw * 3)

def _retry(fn):
    last = None
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(65)
            else:
                last = e
                break
    raise last

def extract_facts_from_text(text, doc_name, page_num):
    if not client:
        st.error("API Key not set.")
        return []
    truncated = text[:MAX_PAGE_CHARS]
    prompt = (
        "Extract the 2 most important facts from the text below.\n"
        "Rules: output max 2 facts. value max 8 words. context max 6 words. "
        "evidence max 12 words must be a substring. id like f1 f2.\n\n"
        "Document: " + doc_name + "  Page: " + str(page_num) + "\n\n"
        "TEXT:\n" + truncated
    )
    try:
        def call():
            return client.chat.completions.create(
                model=MODEL,
                response_model=FactList,
                messages=[
                    {"role": "system", "content": "You are a precise JSON fact-extraction engine."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        response = _retry(call)
        result = []
        for f in response.facts:
            f.source_doc = doc_name
            f.page_num   = page_num
            f.id         = str(uuid.uuid4())[:8]
            result.append(f)
        return result
    except Exception as e:
        st.error("Failed to parse facts: " + str(e))
        return []

def compare_facts(new_facts, existing_facts):
    if not client or not new_facts or not existing_facts:
        return []
    recent = existing_facts[-20:]
    def slim(f):
        return {"id": f.id, "entity": f.entity, "attribute": f.attribute,
                "value": f.value, "context": f.context}
    slim_new = str([slim(f) for f in new_facts])
    slim_ex  = str([slim(f) for f in recent])
    prompt = (
        "Compare NEW facts vs EXISTING facts. Output at most 4 relationships. Skip UNRELATED pairs.\n"
        "Types: CORROBORATES same entity+attribute consistent values. "
        "CONTRADICTS mutually exclusive values. "
        "EXPLAINED_CONTRADICTION context explains difference.\n"
        "reasoning: one sentence max 15 words.\n\n"
        "NEW:\n" + slim_new + "\n\nEXISTING:\n" + slim_ex
    )
    try:
        def call():
            return client.chat.completions.create(
                model=MODEL,
                response_model=RelationshipList,
                messages=[
                    {"role": "system", "content": "You are a precise JSON fact-reconciliation engine."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        response = _retry(call)
        return list(response.relationships)
    except Exception as e:
        st.error("Failed to parse relationships: " + str(e))
        return []

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")
st.title("Fact Knowledge Layer")

if "facts" not in st.session_state:
    st.session_state.facts = []
if "relationships" not in st.session_state:
    st.session_state.relationships = []

with st.sidebar:
    st.header("Configuration")
    if not api_key:
        api_key_input = st.text_input("Groq API Key", type="password")
        if api_key_input:
            client = make_client(api_key_input)
            st.success("API Key set!")
    st.markdown("---")
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        elif not client:
            st.error("Please provide a Groq API Key first.")
        else:
            with st.spinner("Analysing and extracting facts..."):
                for file in uploaded_files:
                    doc_name   = file.name
                    pages_text = extract_text_from_pdf(file)
                    if not pages_text:
                        st.warning("No readable text found in " + doc_name)
                        continue
                    st.subheader("Fact Density Profile: " + doc_name)
                    densities  = [calculate_fact_density(t) for t in pages_text]
                    chart_data = pd.DataFrame(
                        {"Fact Density Score": densities},
                        index=["Page " + str(i+1) for i in range(len(pages_text))]
                    )
                    st.bar_chart(chart_data)
                    n_pages  = min(3, len(pages_text))
                    top_idxs = sorted(range(len(densities)), key=lambda i: densities[i], reverse=True)[:n_pages]
                    selected = ", ".join(str(i+1) for i in sorted(top_idxs))
                    st.caption("Processing pages: " + selected)
                    for i in sorted(top_idxs):
                        new_facts = extract_facts_from_text(pages_text[i], doc_name, i + 1)
                        if new_facts:
                            if st.session_state.facts:
                                rels = compare_facts(new_facts, st.session_state.facts)
                                st.session_state.relationships.extend(rels)
                            st.session_state.facts.extend(new_facts)
            st.success("Processing complete!")
    if st.button("Clear Knowledge Base", use_container_width=True):
        st.session_state.facts         = []
        st.session_state.relationships = []
        st.rerun()

st.header("Knowledge Base")
tab_facts, tab_rels = st.tabs(["Facts", "Relationships"])

with tab_facts:
    if st.session_state.facts:
        df = pd.DataFrame([f.model_dump() for f in st.session_state.facts])
        st.dataframe(df, use_container_width=True)
        st.caption("Total facts: " + str(len(st.session_state.facts)))
    else:
        st.info("No facts extracted yet. Upload PDFs and click Process Documents.")

with tab_rels:
    if st.session_state.relationships:
        id_to_fact = {f.id: f for f in st.session_state.facts}
        def describe_fact(fid):
            f = id_to_fact.get(fid)
            if not f:
                return "[Unknown ID: " + fid + "]"
            ctx = " (" + str(f.context) + ")" if f.context else ""
            return f.entity + " - " + f.attribute + ": " + f.value + ctx + "\n[Evidence: \"" + f.evidence + "\"]"
        rows = []
        for r in st.session_state.relationships:
            rows.append({
                "Relationship": r.relationship,
                "Fact 1":       describe_fact(r.fact1_id),
                "Fact 2":       describe_fact(r.fact2_id),
                "Reasoning":    r.reasoning,
            })
        df_rels = pd.DataFrame(rows)
        def color_rel(val):
            return {"CORROBORATES": "color: green", "CONTRADICTS": "color: red",
                    "EXPLAINED_CONTRADICTION": "color: orange"}.get(val, "")
        st.dataframe(df_rels.style.map(color_rel, subset=["Relationship"]), use_container_width=True)
        st.caption("Total relationships: " + str(len(st.session_state.relationships)))
    else:
        st.info("No relationships found yet.")
