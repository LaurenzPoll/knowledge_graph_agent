from langchain.embeddings import HuggingFaceEmbeddings

from pathlib import Path
from langchain.llms import LlamaCpp

def load_llm() -> LlamaCpp:
    """
    Load and return a LlamaCpp model for extraction and QA.
    """
    model_path = Path(__file__).parent.parent.parent / "models" / "mythomax-l2-13b.Q5_K_M.gguf"
    return LlamaCpp(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=16,
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

def load_embedder():
    """
    Load a HuggingFaceEmbeddings model for embeddings
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")