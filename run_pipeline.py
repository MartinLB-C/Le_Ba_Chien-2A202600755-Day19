import os
from dotenv import load_dotenv
import glob
import random
import json
import re
import hashlib
from tqdm import tqdm
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import pandas as pd
import time

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

MAX_DOCS = 10
RANDOM_SEED = 42

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL")
ALIBABA_MODEL = os.getenv("ALIBABA_MODEL")

if not ALIBABA_BASE_URL or not ALIBABA_MODEL:
    print("Vui lòng cấu hình ALIBABA_BASE_URL và ALIBABA_MODEL trong file .env!")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CACHE_FILE = os.path.join(BASE_DIR, "alibaba_langextract_cache.json")
QA_CACHE_FILE = os.path.join(BASE_DIR, "qwen_qa_cache.json")   

MAX_TRIPLES_PER_CHUNK = 6
MAX_CHARS_PER_CHUNK_FOR_API = 4000
MAX_EXTRACTION_CALLS = 10 # Bắt đầu với 10 để test theo yêu cầu, sau đó có thể tăng lên 30 hoặc 60

# Giai đoạn 3: Load ngẫu nhiên 10 file
def load_documents_random(directory, max_docs=10, seed=42):
    paths = glob.glob(os.path.join(directory, "*.txt"))
    random.seed(seed)
    random.shuffle(paths)

    if max_docs:
        paths = paths[:max_docs]

    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({
                "doc_id": os.path.basename(p),
                "path": p,
                "text": f.read()
            })
    return docs

documents = load_documents_random(DATASET_DIR, max_docs=MAX_DOCS, seed=RANDOM_SEED)
print("Loaded docs:", len(documents))

# Giai đoạn 4: Chunk 10 file này
def chunk_text(text, chunk_size=1200, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks

chunks = []
for d in documents:
    parts = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    for i, text in enumerate(parts):
        chunks.append({
            "chunk_id": f"{d['doc_id']}::ch{i}",
            "doc_id": d["doc_id"],
            "text": text
        })

print("Total chunks:", len(chunks))
chunks_used = chunks[:MAX_EXTRACTION_CALLS]
print("Chunks used for extraction:", len(chunks_used))

# Giai đoạn 5: Dùng LangExtract-style extraction với Alibaba
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=ALIBABA_BASE_URL,
)

def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return text

def build_langextract_style_prompt(text):
    return f"""
You are extracting structured information for a GraphRAG knowledge graph.

Extract factual entity-relation triples from the text.

Return ONLY valid JSON in this schema:
{{
  "triples": [
    {{
      "subject": "",
      "subject_type": "",
      "relation": "",
      "object": "",
      "object_type": "",
      "evidence": ""
    }}
  ]
}}

Rules:
- Maximum {MAX_TRIPLES_PER_CHUNK} triples.
- Extract only explicit facts from the text.
- Do not infer.
- relation must be a short snake_case verb phrase.
- Focus on technology companies, people, products, acquisitions, partnerships, investments, competitors, headquarters, and leadership.
- evidence must be a short phrase copied from the text.

Example:
Text: OpenAI was founded by Sam Altman in 2015.
Output:
{{
  "triples": [
    {{
      "subject": "OpenAI",
      "subject_type": "company",
      "relation": "founded_by",
      "object": "Sam Altman",
      "object_type": "person",
      "evidence": "OpenAI was founded by Sam Altman"
    }},
    {{
      "subject": "OpenAI",
      "subject_type": "company",
      "relation": "founded_in",
      "object": "2015",
      "object_type": "year",
      "evidence": "founded by Sam Altman in 2015"
    }}
  ]
}}

Text:
{text[:MAX_CHARS_PER_CHUNK_FOR_API]}
"""

def extract_triples_alibaba_langextract_style(text):
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_alibaba_key":
        print("API Key is missing, returning empty triples for demo.")
        return []
    
    prompt = build_langextract_style_prompt(text)
    resp = client.chat.completions.create(
        model=ALIBABA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise information extraction engine. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=800,
    )
    content = resp.choices[0].message.content
    content = clean_json_text(content)
    try:
        data = json.loads(content)
        return data.get("triples", [])
    except Exception as e:
        print("JSON parse error:", e)
        print(content[:500])
        return []

# Giai đoạn 6: Thêm cache để không mất quota

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        EXTRACTION_CACHE = json.load(f)
else:
    EXTRACTION_CACHE = {}

def cache_key(text):
    raw = text[:MAX_CHARS_PER_CHUNK_FOR_API]
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(EXTRACTION_CACHE, f, ensure_ascii=False, indent=2)

def extract_triples_cached(text):
    key = cache_key(text)
    if key in EXTRACTION_CACHE:
        return EXTRACTION_CACHE[key]
    triples = extract_triples_alibaba_langextract_style(text)
    EXTRACTION_CACHE[key] = triples
    save_cache()
    return triples

# Giai đoạn 7: Chạy extraction cho chunks
all_triples = []
errors = []
api_calls = 0

for ch in tqdm(chunks_used, desc="Extracting triples"):
    try:
        triples = extract_triples_cached(ch["text"])
        api_calls += 1
    except Exception as e:
        errors.append({
            "chunk_id": ch["chunk_id"],
            "error": repr(e)
        })
        triples = []

    for t in triples:
        if t.get("subject") and t.get("relation") and t.get("object"):
            all_triples.append({
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "subject": str(t["subject"]).strip(),
                "subject_type": str(t.get("subject_type", "")).strip(),
                "relation": str(t["relation"]).strip(),
                "object": str(t["object"]).strip(),
                "object_type": str(t.get("object_type", "")).strip(),
                "evidence": str(t.get("evidence", "")).strip()
            })

print("API calls:", api_calls)
print("Triples:", len(all_triples))
print("Errors:", len(errors))

with open(os.path.join(BASE_DIR, "extracted_triples.json"), "w", encoding="utf-8") as f:
    json.dump(all_triples, f, ensure_ascii=False, indent=2)

# Giai đoạn 8: Clean entity/relation
def normalize_entity(name):
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .,:;()[]{}")
    return name

def normalize_relation(rel):
    rel = str(rel).strip().lower()
    rel = re.sub(r"[^a-z0-9]+", "_", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel

clean_triples = []
for t in all_triples:
    s = normalize_entity(t["subject"])
    o = normalize_entity(t["object"])
    r = normalize_relation(t["relation"])
    if len(s) > 1 and len(o) > 1 and len(r) > 1:
        clean_triples.append({
            **t,
            "subject": s,
            "object": o,
            "relation": r
        })
print("Clean triples:", len(clean_triples))

# Giai đoạn 9: Build graph NetworkX
G = nx.MultiDiGraph()
for t in clean_triples:
    s = t["subject"]
    o = t["object"]
    r = t["relation"]
    G.add_node(s, type=t.get("subject_type", ""))
    G.add_node(o, type=t.get("object_type", ""))
    G.add_edge(
        s,
        o,
        relation=r,
        chunk_id=t["chunk_id"],
        doc_id=t["doc_id"],
        evidence=t.get("evidence", "")
    )
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Giai đoạn 10: Vẽ graph để nộp
plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.35)
nx.draw_networkx_labels(G, pos, font_size=8)
edge_labels = {}
for u, v, data in G.edges(data=True):
    edge_labels[(u, v)] = data.get("relation", "")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "knowledge_graph.png"), dpi=200)

# Giai đoạn 11: Flat RAG baseline bằng TF-IDF
class LocalTfidfIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.items = []
        self.matrix = None

    def build(self, items):
        self.items = items
        texts = [it["text"] for it in items]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=4):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        idxs = scores.argsort()[::-1][:k]
        return [
            {
                **self.items[i],
                "score": float(scores[i])
            }
            for i in idxs
        ]

flat_index = LocalTfidfIndex()
flat_index.build(chunks)

# Giai đoạn 12: GraphRAG query 2-hop
def link_entity(query, G):
    q = query.lower()
    nodes = list(G.nodes)
    for n in nodes:
        if n.lower() in q:
            return n
    matches = get_close_matches(query, nodes, n=1, cutoff=0.35)
    return matches[0] if matches else None

def graph_context(query, hops=2):
    seed = link_entity(query, G)
    if not seed:
        return ""
    visited = {seed}
    frontier = {seed}
    lines = []
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for _, v, data in G.out_edges(node, data=True):
                rel = data.get("relation", "")
                ev = data.get("evidence", "")
                lines.append(f"{node} --{rel}--> {v}. Evidence: {ev}")
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
            for u, _, data in G.in_edges(node, data=True):
                rel = data.get("relation", "")
                ev = data.get("evidence", "")
                lines.append(f"{u} --{rel}--> {node}. Evidence: {ev}")
                if u not in visited:
                    visited.add(u)
                    next_frontier.add(u)
        frontier = next_frontier
    return "\n".join(lines)

# Giai đoạn 13: Trả lời bằng Qwen

if os.path.exists(QA_CACHE_FILE):
    with open(QA_CACHE_FILE, "r", encoding="utf-8") as f:
        QA_CACHE = json.load(f)
else:
    QA_CACHE = {}

def save_qa_cache():
    with open(QA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(QA_CACHE, f, ensure_ascii=False, indent=2)

def qwen_answer(question, context):
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_alibaba_key":
        return "Simulated answer since API Key is missing."
        
    qa_key = hashlib.md5((question + context[:3000]).encode("utf-8")).hexdigest()
    if qa_key in QA_CACHE:
        return QA_CACHE[qa_key]

    prompt = f"""
Answer the question using only the context below.
If the context is insufficient, say the data is insufficient.

Context:
{context[:3000]}

Question:
{question}

Answer:
"""
    try:
        resp = client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer strictly from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=300,
        )
        ans = resp.choices[0].message.content
        QA_CACHE[qa_key] = ans
        save_qa_cache()
        return ans
    except Exception as e:
        print("QA Error:", e)
        return "Error calling API."


def graphrag_answer(question):
    ctx = graph_context(question, hops=2)
    if not ctx.strip():
        return "Không tìm thấy entity phù hợp trong graph.", ctx
    return qwen_answer(question, ctx), ctx

def flat_rag_answer(question):
    hits = flat_index.search(question, k=4)
    ctx = "\n\n".join(
        f"[{h['chunk_id']}]\n{h['text'][:900]}"
        for h in hits
    )
    return qwen_answer(question, ctx), ctx

# Giai đoạn 14: Benchmark theo yêu cầu lab
benchmark_questions = [
    "What entities are connected to Microsoft?",
    "What entities are connected to OpenAI?",
    "What products are connected to Google?",
    "Which companies are connected through partnerships?",
    "Which companies are connected through acquisitions?",
    "Which people are connected to technology companies?",
    "What competitors are mentioned?",
    "Which companies are connected to AI products?",
    "What headquarters or locations are mentioned?",
    "What investments are mentioned?",
    "What is connected to NVIDIA?",
    "What is connected to Tesla?",
    "What is connected to Amazon?",
    "What is connected to Apple?",
    "What is connected to Meta?",
    "What multi-hop relation can be found around Microsoft?",
    "What multi-hop relation can be found around Google?",
    "What multi-hop relation can be found around AI?",
    "Which products are connected to companies?",
    "Which company relationships are most important in the graph?"
]

rows = []
for q in benchmark_questions:
    t0 = time.time()
    flat_ans, flat_ctx = flat_rag_answer(q)
    flat_time = time.time() - t0

    t0 = time.time()
    graph_ans, graph_ctx = graphrag_answer(q)
    graph_time = time.time() - t0

    rows.append({
        "question": q,
        "flat_answer": flat_ans,
        "graph_answer": graph_ans,
        "flat_time": flat_time,
        "graph_time": graph_time,
        "flat_context_chars": len(flat_ctx),
        "graph_context_chars": len(graph_ctx),
        "winner": "",
        "note": ""
    })

benchmark_df = pd.DataFrame(rows)
benchmark_df.to_csv(os.path.join(BASE_DIR, "benchmark_flat_vs_graphrag.csv"), index=False)
print("Saved benchmark to benchmark_flat_vs_graphrag.csv")
