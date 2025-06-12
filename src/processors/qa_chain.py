import numpy as np
import difflib
from typing import List, Dict, Optional, Tuple
import streamlit as st


def normalize_entity(question: str, candidates: List[str]) -> Optional[str]:
    """
    Identify if the users question mentions one of the node (candidate) names allowing for minor typos via fuzzy matched_nameing. 
    The question is split into words then compare each word against each node name, and return the best matched_nameing
    node if its similarity is above the cutoff of 0.6, else None.
    """
    best_matched_name: Optional[str] = None
    best_score = 0.0

    words = question.replace("?", "").split()

    for word in words:
        for candidate in candidates:
            similarity = difflib.SequenceMatched_nameer(
                None, word.lower(), candidate.lower()
            ).ratio()
            if similarity > best_score and similarity > 0.6:
                best_score, best_matched_name = similarity, candidate

    return best_matched_name


def answer_question(
    elements: List[Dict],
    question: str,
    llm,
    embedder, # HuggingFaceEmbeddings instance
    top_k: int = 5
) -> str:
    """
    1) Extract triples from Cytoscape elements
    2) Embed all triples once (cached)
    3) Optionally filter to a single entity's facts
    4) Embed the question
    5) Rank triples by cosine similarity
    6) Format a bullet-list context and call the LLM
    Returns: (answer, bullet_list)
    """

    # 1) Pull out (subject, predicate, object) triples and make text lines
    triples_list, text_representations = extract_triples(elements)

    if not triples_list:
        return "No facts available to answer your question"

    # 2) Embed all triple texts once and cache the result
    text_tuple: Tuple[str, ...] = tuple(text_representations)
    embeddings: np.ndarray = _embed_all_triples(text_tuple, embedder)

    # 3) If user mentioned a particular entity, filter to its facts
    triples_list, embeddings = filter_triples_by_entity(question, triples_list, embeddings)
            
    # 4) Embed the user's question
    question_embedding = embed_string(question, embedder)

    # 5) Compute cosine similarity and select top_k triples
    selected_triples = rank_and_select(top_k, triples_list, embeddings, question_embedding)

    # 6) Build a bullet-list context and call the LLM
    bullet_list, prompt = build_prompt(question, selected_triples)
    response = llm(
        prompt,
        max_tokens=200,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        stop=["User:", "\n\n"],
    )

    return response.strip(), bullet_list


def extract_triples(elements):
    """
    Pull out (subject, predicate, object) triples and make text lines
    """
    triples_list: List[Tuple[str,str,str]] = []
    text_representations: List[str] = []

    for element in elements:
        data = element.get("data", {})
        subject = data.get("source")
        predicate = data.get("label")
        object_ = data.get("target")

        if subject and predicate and object_:
            triples_list.append((subject, predicate, object_))
            text_representations.append(f"{subject} | {predicate} | {object_}")

    return triples_list, text_representations


@st.cache_data(show_spinner=False)
def _embed_all_triples(texts: Tuple[str, ...], embedder) -> np.ndarray:
    """
    Embed all triple texts once and cache the result
    """
    return np.array(embedder.embed_documents(list(texts)))


def filter_triples_by_entity(question, triples_list, embeddings):
    """
    If user mentioned a particular entity, filter to its facts
    """
    node_names = list({subject for subject,_,_ in triples_list} | {object_ for _,_,object_ in triples_list})
    matched_name = normalize_entity(question, node_names)
    if matched_name:
        # build filter_mask for triples involving that entity
        filter_mask = [
            (subject == matched_name or object_ == matched_name)
            for subject, _, object_ in triples_list
        ]
        if any(filter_mask):
            # Subset triples, texts, and embeddings
            triples_list = [
                t for t, keep in zip(triples_list, filter_mask) if keep
            ]
            text_representations = [
                txt for txt, keep in zip(text_representations, filter_mask) if keep
            ]
            embeddings = embeddings[np.array(filter_mask)]

    return triples_list, embeddings


def embed_string(question, embedder) -> np.ndarray:
    """
    Embed the user's question
    """
    question_embedding = np.array(embedder.embed_query(question))
    return question_embedding


def rank_and_select(top_k, triples_list, embeddings, question_embedding) -> list:
    """
    Compute cosine similarity and select top_k triples
    """
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    norms[norms == 0] = 1e-8
    similarities = (embeddings @ question_embedding) / norms
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    selected_triples = [triples_list[i] for i in top_indices]
    return selected_triples


def build_prompt(question, selected_triples) -> Tuple[str, str]:
    """
    Build a bullet-list context and call the LLM
    """
    bullet_list = "\n".join(
        f"- **{subject}** {predicate} **{object_}**"
        for subject, predicate, object_ in selected_triples
    )

    prompt = (
        "You are a fact-based QA assistant.\n"
        "Use only the following information to answer the question. Do not add any extra information:\n\n"
        f"{bullet_list}\n\n"
        f"User: {question}\nAssistant:"
    )
    print("full prompt: ", prompt)
    return bullet_list,prompt