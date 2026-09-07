# Fact Knowledge Layer

A knowledge graph system that extracts meaningful facts from PDFs, links them to evidence, and identifies relationships (corroborations, contradictions, and contextual explanations).

## Setup and Run Instructions

### Prerequisites
- Python 3.10+
- A Groq API key

### Setup
1. Clone the repository and navigate into it.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
   *Alternatively, you can provide the API key directly in the UI sidebar.*

### Running the App
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL in your browser.
3. Use the sidebar to upload PDFs and process them.

## Video Demo
[Insert Link to Video Demo here - <3 mins]

## Approach

**Architecture:**
- **UI & Flow Control:** Built with Streamlit for a simple, responsive interface allowing iterative uploads.
- **PDF Extraction:** `pypdf` is used to quickly scrape raw text.
- **LLM Engine:** Groq API using `llama-3.3-70b-versatile` via the `instructor` library to guarantee structured Pydantic schema outputs.
- **Knowledge Layer:** Kept in session state (easily extendable to a graph database like Neo4j) representing nodes (Facts) and edges (Relationships).

**Important Decisions & Trade-offs:**
1. **Fact Schema:** Defined as `Entity`, `Attribute`, `Value`, `Context`, and `Evidence`. The explicit `Context` parameter (time, scope) provides a mechanism for the reconciler to discern whether a contradiction is genuine or context-based.
2. **Two-Step Processing:**
   - *Extraction* is performed page-by-page. This ensures we don't exceed token limits and ground facts explicitly to their source document and page.
   - *Reconciliation* is performed collectively. As new facts are discovered, they are compared with existing facts to find Corroborations, Contradictions, or Explained Contradictions.
3. **Trade-off:** Comparing every new fact to every existing fact ($O(N^2)$) works for prototypes but scales poorly for thousands of facts. I chose this for accuracy in the prototype.

## The Four Cases Demonstrated

Included in the `data/` folder are 4 dummy documents representing financial reports.
1. **Corroborated Fact:** `Doc2` and `Doc3` both confirm the Q2 revenue is $8M.
2. **Genuine Contradiction:** `Doc1/Doc2` claim Q1 revenue was $10M, while the later `Doc3_Audit_Report` claims it was $15M.
3. **Apparent Contradiction Explained:** `Doc1` names John Doe as CEO, while `Doc2` names Jane Smith. The system reconciles this as an `EXPLAINED_CONTRADICTION` due to the context of time (Q1 vs. Q2).
4. **Extraction/Reasoning Failure:**
   - *Failure Encountered:* Initially, the LLM flagged "$10M" and "$10,000,000" as a contradiction due to string mismatch.
   - *How it was Handled:* I updated the reconciliation prompt instructions to prioritize semantic unit equivalence and handle apparent numeric formatting differences as `EXPLAINED_CONTRADICTION` or `CORROBORATES`. Also, enforcing the `Context` field significantly improved temporal alignment (e.g., distinguishing Q1 vs. Q2).

## Limitations and Next Steps

**Limitations:**
- **Scaling Reconciliation:** The $O(N^2)$ comparison between new and existing facts hits context limits quickly on large knowledge bases.
- **Tabular Data:** Basic PDF text extraction struggles with complex tables and unstructured layouts.
- **Ephemeral Storage:** The knowledge base currently lives in Streamlit session state and resets on refresh.

**Next Steps (What I would build next):**
- **Vector Search / Graph Database:** Implement Neo4j to store the facts. Before comparing a new fact, query Neo4j for nodes with similar entity names or embeddings to limit the context sent to the LLM.
- **Vision-based PDF Parsing:** Use multimodal models (like Gemini 1.5 Pro) or specialized OCR tools (like Unstructured or LlamaParse) to process PDFs visually, correctly extracting facts from dense tables and charts.
- **Human-in-the-Loop:** Add a UI layer to allow users to manually approve, reject, or edit extracted facts and relationships before they are committed to the graph.

## Additional Notes
I created a script (`generate_pdfs.py`) to generate sample documents for demonstration purposes to cover all edge cases perfectly. Feel free to use your own PDFs!
