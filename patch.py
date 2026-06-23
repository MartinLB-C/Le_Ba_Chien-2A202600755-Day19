import json
import os

# --- Patch run_pipeline.py ---
pipeline_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\run_pipeline.py"
with open(pipeline_file, "r", encoding="utf-8") as f:
    code = f.read()

# Replace config
code = code.replace(
    'ALIBABA_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"\nALIBABA_MODEL = "qwen3-vl-flash"',
    'ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")\nALIBABA_MODEL = os.getenv("ALIBABA_MODEL", "qwen3-vl-flash")\n\nBASE_DIR = os.path.dirname(os.path.abspath(__file__))\nDATASET_DIR = os.path.join(BASE_DIR, "dataset")\nCACHE_FILE = os.path.join(BASE_DIR, "alibaba_langextract_cache.json")\nQA_CACHE_FILE = os.path.join(BASE_DIR, "qwen_qa_cache.json")'
)
code = code.replace('DATASET_DIR = "dataset"\n', "")
code = code.replace('CACHE_FILE = "alibaba_langextract_cache.json"\n', "")

# Replace paths in save files
code = code.replace(
    'with open("extracted_triples.json", "w", encoding="utf-8") as f:',
    'with open(os.path.join(BASE_DIR, "extracted_triples.json"), "w", encoding="utf-8") as f:'
)
code = code.replace(
    'plt.savefig("knowledge_graph.png", dpi=200)',
    'plt.savefig(os.path.join(BASE_DIR, "knowledge_graph.png"), dpi=200)'
)
code = code.replace(
    'benchmark_df.to_csv("benchmark_flat_vs_graphrag.csv", index=False)',
    'benchmark_df.to_csv(os.path.join(BASE_DIR, "benchmark_flat_vs_graphrag.csv"), index=False)'
)

# Add QA cache
qa_cache_code = """
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

    prompt = f\"\"\"
Answer the question using only the context below.
If the context is insufficient, say the data is insufficient.

Context:
{context[:3000]}

Question:
{question}

Answer:
\"\"\"
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
"""

# We replace the original qwen_answer with the new cached version
old_qwen = """def qwen_answer(question, context):
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_alibaba_key":
        return "Simulated answer since API Key is missing."
        
    prompt = f\"\"\"
Answer the question using only the context below.
If the context is insufficient, say the data is insufficient.

Context:
{context[:3000]}

Question:
{question}

Answer:
\"\"\"
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
    return resp.choices[0].message.content"""

code = code.replace(old_qwen, qa_cache_code)

with open(pipeline_file, "w", encoding="utf-8") as f:
    f.write(code)

# --- Patch graphrag_alibaba.ipynb ---
nb_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\graphrag_alibaba.ipynb"
with open(nb_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3: Config
nb["cells"][3]["source"] = [
    "import random\n",
    "import os\n",
    "\n",
    "DATASET_DIR = \"dataset\"\n",
    "MAX_DOCS = 10\n",
    "RANDOM_SEED = 42\n",
    "\n",
    "CHUNK_SIZE = 1200\n",
    "CHUNK_OVERLAP = 150\n",
    "\n",
    "GRAPH_BACKEND = \"networkx\"\n",
    "EXTRACTION_BACKEND = \"langextract\"\n",
    "\n",
    "ALIBABA_BASE_URL = os.getenv(\"ALIBABA_BASE_URL\", \"https://dashscope-intl.aliyuncs.com/compatible-mode/v1\")\n",
    "ALIBABA_MODEL = os.getenv(\"ALIBABA_MODEL\", \"qwen3-vl-flash\")\n",
    "\n",
    "MAX_TRIPLES_PER_CHUNK = 6\n",
    "MAX_CHARS_PER_CHUNK_FOR_API = 4000\n",
    "MAX_EXTRACTION_CALLS = 30"
]

# Add QA Cache into Cell 19 (which contains qwen_answer)
old_qwen_nb = [
    "def qwen_answer(question, context):\n",
    "    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == \"your_alibaba_key\":\n",
    "        return \"C\\u1ea7n c\\u00f3 DashScope API Key \\u0111\\u1ec3 g\\u1ecdi Qwen API.\"\n",
    "        \n",
    "    prompt = f\"\"\"\n",
    "Answer the question using only the context below.\n",
    "If the context is insufficient, say the data is insufficient.\n",
    "\n",
    "Context:\n",
    "{context[:3000]}\n",
    "\n",
    "Question:\n",
    "{question}\n",
    "\n",
    "Answer:\n",
    "\"\"\"\n",
    "\n",
    "    resp = client.chat.completions.create(\n",
    "        model=ALIBABA_MODEL,\n",
    "        messages=[\n",
    "            {\n",
    "                \"role\": \"system\",\n",
    "                \"content\": \"Answer strictly from the provided context.\"\n",
    "            },\n",
    "            {\n",
    "                \"role\": \"user\",\n",
    "                \"content\": prompt\n",
    "            }\n",
    "        ],\n",
    "        temperature=0,\n",
    "        max_tokens=300,\n",
    "    )\n",
    "\n",
    "    return resp.choices[0].message.content\n",
    "\n",
    "def graphrag_answer(question):\n",
    "    ctx = graph_context(question, hops=2)\n",
    "\n",
    "    if not ctx.strip():\n",
    "        return \"Kh\\u00f4ng t\\u00ecm th\\u1ea5y entity ph\\u00f9 h\\u1ee3p trong graph.\", ctx\n",
    "\n",
    "    return qwen_answer(question, ctx), ctx\n",
    "\n",
    "def flat_rag_answer(question):\n",
    "    hits = flat_index.search(question, k=4)\n",
    "\n",
    "    ctx = \"\\n\\n\".join(\n",
    "        f\"[{h['chunk_id']}]\\n{h['text'][:900]}\"\n",
    "        for h in hits\n",
    "    )\n",
    "\n",
    "    return qwen_answer(question, ctx), ctx"
]

new_qwen_nb = [
    "import hashlib\n",
    "QA_CACHE_FILE = \"qwen_qa_cache.json\"\n",
    "if os.path.exists(QA_CACHE_FILE):\n",
    "    with open(QA_CACHE_FILE, \"r\", encoding=\"utf-8\") as f:\n",
    "        QA_CACHE = json.load(f)\n",
    "else:\n",
    "    QA_CACHE = {}\n",
    "\n",
    "def save_qa_cache():\n",
    "    with open(QA_CACHE_FILE, \"w\", encoding=\"utf-8\") as f:\n",
    "        json.dump(QA_CACHE, f, ensure_ascii=False, indent=2)\n",
    "\n",
    "def qwen_answer(question, context):\n",
    "    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == \"your_alibaba_key\":\n",
    "        return \"C\\u1ea7n c\\u00f3 DashScope API Key \\u0111\\u1ec3 g\\u1ecdi Qwen API.\"\n",
    "        \n",
    "    qa_key = hashlib.md5((question + context[:3000]).encode(\"utf-8\")).hexdigest()\n",
    "    if qa_key in QA_CACHE:\n",
    "        return QA_CACHE[qa_key]\n",
    "\n",
    "    prompt = f\"\"\"\n",
    "Answer the question using only the context below.\n",
    "If the context is insufficient, say the data is insufficient.\n",
    "\n",
    "Context:\n",
    "{context[:3000]}\n",
    "\n",
    "Question:\n",
    "{question}\n",
    "\n",
    "Answer:\n",
    "\"\"\"\n",
    "\n",
    "    try:\n",
    "        resp = client.chat.completions.create(\n",
    "            model=ALIBABA_MODEL,\n",
    "            messages=[\n",
    "                {\n",
    "                    \"role\": \"system\",\n",
    "                    \"content\": \"Answer strictly from the provided context.\"\n",
    "                },\n",
    "                {\n",
    "                    \"role\": \"user\",\n",
    "                    \"content\": prompt\n",
    "                }\n",
    "            ],\n",
    "            temperature=0,\n",
    "            max_tokens=300,\n",
    "        )\n",
    "        ans = resp.choices[0].message.content\n",
    "        QA_CACHE[qa_key] = ans\n",
    "        save_qa_cache()\n",
    "        return ans\n",
    "    except Exception as e:\n",
    "        print(\"QA Error:\", e)\n",
    "        return \"Error calling API.\"\n",
    "\n",
    "def graphrag_answer(question):\n",
    "    ctx = graph_context(question, hops=2)\n",
    "\n",
    "    if not ctx.strip():\n",
    "        return \"Kh\\u00f4ng t\\u00ecm th\\u1ea5y entity ph\\u00f9 h\\u1ee3p trong graph.\", ctx\n",
    "\n",
    "    return qwen_answer(question, ctx), ctx\n",
    "\n",
    "def flat_rag_answer(question):\n",
    "    hits = flat_index.search(question, k=4)\n",
    "\n",
    "    ctx = \"\\n\\n\".join(\n",
    "        f\"[{h['chunk_id']}]\\n{h['text'][:900]}\"\n",
    "        for h in hits\n",
    "    )\n",
    "\n",
    "    return qwen_answer(question, ctx), ctx"
]

nb["cells"][19]["source"] = new_qwen_nb

with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patch successful!")
