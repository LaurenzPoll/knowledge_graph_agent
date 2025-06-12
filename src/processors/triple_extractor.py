from typing import List, Dict
from langchain.schema import Document
from langchain.llms import LlamaCpp

SYSTEM = """You are an information-extraction assistant.
Your only job is to pull out valid subject | predicate | object triples from the text that follows.
**Only** use facts explicitly stated in that text. Do **not** use any external or world knowledge.
Output one triple per line in the format:
  subject | predicate | object
No other text or commentary.
When you see a verb phrase followed by a prepositional phrase starting with “during”, “in”, “on”, or “at”, 
treat that prepositional phrase as the object and do not include it in the predicate.
"""

FEW_SHOT = """
User: Alice and Bob co-authored a research paper on natural language processing, which was published in 2021.
Assistant:
Alice | co-authored | a research paper on natural language processing
Bob | co-authored | a research paper on natural language processing
a research paper on natural language processing | was published in | 2021

User: John works at Acme Corp and lives in Paris.
Assistant:
John | works at | Acme Corp
John | lives in | Paris

User: Sarah, the founder of TechStart, delivered the keynote in Berlin. She also launched a new product.
Assistant:
Sarah | is the founder of | TechStart
Sarah | delivered | the keynote in Berlin
Sarah | launched | a new product

User: The conference rose to prominence during the Industrial Revolution.
Assistant:
The conference | rose to prominence | the Industrial Revolution
"""

def extract_triples(docs: List[Document], llm: LlamaCpp) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    for doc in docs:
        # assemble prompt
        user_block = f"User: {doc.page_content}\nAssistant:\n"
        prompt = "\n\n".join([SYSTEM, FEW_SHOT, user_block])
        print("PROMPT >>>\n", prompt)

        # Call with strict decoding and two‐token stops
        response = llm(
            prompt,
            max_tokens=1250,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.1,
            stop=["\n\n","User:"],
        )
        raw = response.strip()

        # parse each “subject|predicate|object” line
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) == 3:
                subject, predicate, object = parts
                triples.append({"subject": subject, "predicate": predicate, "object": object})

    return triples