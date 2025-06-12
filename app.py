# ──────────────────────────────────────────────────────────────────────────────
# Prevent Streamlit’s watcher from crashing on torch.classes
import torch.classes
class _DummyPath:
    def __init__(self):
        # the watcher does: list(module.__path__._path)
        self._path = []
# Attach a fake __path__ with a ._path attribute
torch.classes.__path__ = _DummyPath()
# ──────────────────────────────────────────────────────────────────────────────


from pathlib import Path
import streamlit as st
from st_cytoscape import cytoscape

from src.loaders.document_loader import load_documents_and_chunk_sentences
from src.model.load_model import load_llm, load_embedder
from src.processors.triple_extractor import extract_triples
from src.processors.graph_builder import build_graph
from src.processors.qa_chain import answer_question
from src.utils.file_utils import save_uploaded_files, clear_uploads_dir

demo_triples = [
    # — Space & NASA
    {"subject": "Apollo 11",          "predicate": "launched on",             "object": "July 16, 1969"},
    {"subject": "Apollo 11",          "predicate": "landed on",               "object": "July 20, 1969"},
    {"subject": "Apollo 11",          "predicate": "returned to Earth on",    "object": "July 24, 1969"},
    {"subject": "Neil Armstrong",     "predicate": "was commander of",        "object": "Apollo 11"},
    {"subject": "Buzz Aldrin",        "predicate": "was lunar module pilot of","object": "Apollo 11"},
    {"subject": "Michael Collins",    "predicate": "was command module pilot of","object": "Apollo 11"},
    {"subject": "Saturn V",           "predicate": "launched",                "object": "Apollo 11"},
    {"subject": "NASA",               "predicate": "operated",                "object": "Apollo 11"},
    {"subject": "Wernher von Braun",  "predicate": "designed",                "object": "Saturn V"},
    {"subject": "Saturn V",           "predicate": "first launched on",       "object": "November 9, 1967"},
    {"subject": "Saturn V",           "predicate": "last launched on",        "object": "May 14, 1973"},

    # — History: Napoleon Bonaparte
    {"subject": "Napoleon Bonaparte", "predicate": "born on",                "object": "August 15, 1769"},
    {"subject": "Napoleon Bonaparte", "predicate": "crowned emperor on",     "object": "December 2, 1804"},
    {"subject": "Napoleon Bonaparte", "predicate": "exiled to",              "object": "Elba"},
    {"subject": "Napoleon Bonaparte", "predicate": "exiled to",              "object": "Saint Helena"},
    {"subject": "Napoleon Bonaparte", "predicate": "died on",                "object": "May 5, 1821"},

    # — History: Julius Caesar
    {"subject": "Julius Caesar",      "predicate": "born on",                "object": "July 12, 100 BC"},
    {"subject": "Julius Caesar",      "predicate": "crossed",                "object": "the Rubicon"},
    {"subject": "Julius Caesar",      "predicate": "was dictator of",        "object": "Roman Republic"},
    {"subject": "Julius Caesar",      "predicate": "assassinated on",        "object": "March 15, 44 BC"},
    {"subject": "Julius Caesar",      "predicate": "assassinated in",        "object": "Rome"},

    # — Literature: William Shakespeare
    {"subject": "William Shakespeare","predicate": "born on",                "object": "April 26, 1564"},
    {"subject": "William Shakespeare","predicate": "born in",                "object": "Stratford-upon-Avon"},
    {"subject": "William Shakespeare","predicate": "died on",                "object": "April 23, 1616"},
    {"subject": "William Shakespeare","predicate": "wrote",                  "object": "Hamlet"},
    {"subject": "William Shakespeare","predicate": "wrote",                  "object": "Julius Caesar"},
    {"subject": "William Shakespeare","predicate": "wrote",                  "object": "Henry V"},

    # — Literature ↔ History overlap
    {"subject": "Henry V",            "predicate": "was King of",            "object": "England"},
    {"subject": "Henry V",            "predicate": "died on",                "object": "August 31, 1422"},
    {"subject": "Roman Republic",     "predicate": "capital was",            "object": "Rome"},

    # — Geography ↔ Literature overlap
    {"subject": "Stratford-upon-Avon", "predicate": "is in",                 "object": "England"},

    # — Music: The Beatles
    {"subject": "The Beatles",        "predicate": "formed in",              "object": "Liverpool"},
    {"subject": "The Beatles",        "predicate": "released",               "object": "Abbey Road"},
    {"subject": "Abbey Road",         "predicate": "released on",            "object": "September 26, 1969"},
    {"subject": "John Lennon",        "predicate": "was member of",          "object": "The Beatles"},
    {"subject": "Paul McCartney",     "predicate": "was member of",          "object": "The Beatles"},
    {"subject": "George Harrison",    "predicate": "was member of",          "object": "The Beatles"},
    {"subject": "Ringo Starr",        "predicate": "was member of",          "object": "The Beatles"},
    {"subject": "John Lennon",        "predicate": "born in",                "object": "Liverpool"},

    # — Science: CRISPR
    {"subject": "Jennifer Doudna",    "predicate": "co-invented",            "object": "CRISPR-Cas9"},
    {"subject": "CRISPR-Cas9",        "predicate": "used for",               "object": "gene editing"},
    {"subject": "Jennifer Doudna",    "predicate": "won",                    "object": "Nobel Prize in Chemistry 2020"},
]


# directory for user uploaded documents
UPLOAD_DIR = Path("uploads")

def main():
    st.set_page_config(
        page_title="Knowledge Graph Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📚 Knowledge Graph Agent")

    # load models once
    llm = load_llm()
    embedder = load_embedder()

    elements = st.session_state.get("cyto_elements")
    G = st.session_state.get("graph")

    with st.sidebar:
        if elements and G is not None:
            with st.form("query_form", clear_on_submit=True):
                st.markdown("## Ask a Question")
                query = st.text_input("Type your question about this corpus and hit enter")
                submitted = st.form_submit_button("Ask")

            if submitted and query:
                with st.spinner("Retrieving answer…"):
                    answer, context = answer_question(elements, query, llm, embedder)
                    st.session_state["query"] = query
                    st.session_state["answer"] = answer
                    st.session_state["context"] = context

            if st.session_state.get("query"):
                st.subheader(st.session_state["query"])
            if st.session_state.get("answer"):
                st.subheader("Answer")
                st.write(st.session_state["answer"])
            if st.session_state.get("context"):
                st.subheader("Generated with the context:")
                st.markdown(st.session_state["context"])

        else:
            # No graph yet, show upload option & build graph button
            st.header("1) Upload Documents")
            files = st.file_uploader(
                "Upload one or more documents (txt, md, pdf, docx), they may be about any domain",
                type=["txt", "md", "pdf", "docx"],
                accept_multiple_files=True
            )
            use_demo = st.checkbox("Use demo graph instead of LLM triples", value=True)

            if st.button("2) Build Knowledge Graph"):
                if use_demo:
                    triples = demo_triples
                    st.write("🔍 Using demo triples:", len(triples))
                    st.info("Building graph…")
                    G, elements = build_graph(triples)
                    st.session_state.graph = G
                    st.session_state.cyto_elements = elements
                    st.success("✅ Knowledge graph built!")
                elif not files:
                    st.error("▶️ Please upload at least one file or ZIP.")
                else:
                    with st.spinner("Saving and extracting uploads…"):
                        clear_uploads_dir(UPLOAD_DIR)
                        docs_path = save_uploaded_files(files, UPLOAD_DIR)

                    st.info("Loading documents..")
                    docs = load_documents_and_chunk_sentences(docs_path)
                    st.write("🔍 Raw documents loaded:", len(docs))

                    st.info("Extracting triples..")
                    triples = extract_triples(docs, llm)
                    st.write("🔍 Triples extracted:", len(triples))

                    st.info("Building graph..")
                    G, elements = build_graph(triples)
                    st.session_state.graph = G
                    st.session_state.cyto_elements = elements
                    st.success("✅ Knowledge graph built!")


    def build_focus_styles(G, click_payload):
        styles = []
        if isinstance(click_payload, dict) and click_payload.get("nodes"):
            nid = click_payload["nodes"][0]
            # highlight clicked node
            styles.append({
                "selector": f'node[id = "{nid}"]',
                "style": {"opacity": 1.0, "background-color": "purple", "font-size": 12}
            })

            # highlight neighbors
            preds = set(G.predecessors(nid))
            succs = set(G.successors(nid))
            for nbr in preds | succs:
                styles.append({
                    "selector": f'node[id = "{nbr}"]',
                    "style": {"opacity": 0.9, "background-color": "#2ca02c"}
                })

            # highlight edges
            for source in preds:
                styles.append({
                    "selector": f'edge[source = "{source}"][target = "{nid}"]',
                    "style": {"opacity": 1.0, "line-color": "#f90", "width": 3}
                })
            for target in succs:
                styles.append({
                    "selector": f'edge[source = "{nid}"][target = "{target}"]',
                    "style": {"opacity": 1.0, "line-color": "#f90", "width": 3}
                })
        return styles

    elements = st.session_state.get("cyto_elements")
    G        = st.session_state.get("graph")

    if elements and G is not None:
        base_stylesheet = [
            { "selector": "node", "style": {
                "opacity": 0.4, "label": "data(label)",
                "text-valign": "center", "text-halign": "center",
                "shape": "roundrectangle", "padding": 5,
                "font-size": 10, "text-wrap": "wrap",
                "text-max-width": 100,
                "background-color": "#1f77b4",
                "text-background-color": "#fff",
                "text-background-opacity": 0.8,
            }},
            { "selector": "edge", "style": {
                "opacity": 0.2, "label": "data(label)",
                "font-size": 8, "text-rotation": "autorotate",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#777",
            }},
        ]
        
        prev_click = st.session_state.get("kg_click")
        print("🔹 prev_click:", prev_click)

        # build stylesheet from whichever node was active last run
        focus_styles = build_focus_styles(G, prev_click)
        full_styles  = base_stylesheet + focus_styles

        # cytoscape for graph
        clicked = cytoscape(
            elements=elements,
            stylesheet=full_styles,
            layout={
                "name": "breadthfirst",
                "directed": True,
                "circle": False,
                "spacingFactor": 1.5,
            },
            height="500px",
            width="100%",
            key="kg_viz",
        )

        print("clicked payload:", clicked)
        if isinstance(clicked, dict) and clicked.get("nodes"):
            new_click = clicked
        else:
            new_click = None
        print("new_click:", new_click)

        reran_flag = st.session_state.get("_reran_for_click", False)
        print("reran_flag before:", reran_flag)

        # 4) If it changed, and we haven’t already rerun for this click, do so just once
        if new_click != prev_click and new_click is not None and not reran_flag:
            print("Detected change, setting kg_click & rerunning…")
            st.session_state["kg_click"] = new_click
            st.session_state["_reran_for_click"] = True
            st.rerun()

        # clear rerun flag so new clicks can trigger again
        if reran_flag:
            print("Clearing rerun state")
            del st.session_state["_reran_for_click"]

if __name__ == "__main__":
    main()