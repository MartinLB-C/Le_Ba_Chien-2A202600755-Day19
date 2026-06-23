import json

# --- Patch run_pipeline.py ---
pipeline_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\run_pipeline.py"
with open(pipeline_file, "r", encoding="utf-8") as f:
    code = f.read()

old_benchmark_code = """# Giai đoạn 14: Benchmark theo yêu cầu lab
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
]"""

new_benchmark_code = """# Giai đoạn 14: Benchmark theo yêu cầu lab
# Tự động tạo 20 câu hỏi dựa trên các entities có thật trong đồ thị (G)
top_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]]
if not top_nodes:
    top_nodes = ["AI", "Technology", "Company"] # Fallback nếu graph rỗng

templates = [
    "What entities are connected to {}?",
    "What products are connected to {}?",
    "Which companies are connected through partnerships with {}?",
    "Which companies are connected through acquisitions related to {}?",
    "Which people are connected to {}?",
    "What competitors are mentioned for {}?",
    "Which companies are connected to AI products of {}?",
    "What headquarters or locations are mentioned for {}?",
    "What investments are mentioned for {}?",
    "What is connected to {}?",
    "What multi-hop relation can be found around {}?",
    "Which products are connected to {}?",
    "What are the most important relationships for {} in the graph?",
    "Describe the ecosystem around {}.",
    "What role does {} play in this context?",
    "Who does {} interact with?",
    "What technologies are linked to {}?",
    "How does {} influence others?",
    "What is the main focus of {}?",
    "Summarize the connections of {}."
]

benchmark_questions = []
for i in range(20):
    node = top_nodes[i % len(top_nodes)]
    benchmark_questions.append(templates[i].format(node))"""

code = code.replace(old_benchmark_code, new_benchmark_code)

with open(pipeline_file, "w", encoding="utf-8") as f:
    f.write(code)


# --- Patch graphrag_alibaba.ipynb ---
nb_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\graphrag_alibaba.ipynb"
with open(nb_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 21 is the benchmark generation
nb_source = [
    "import pandas as pd\n",
    "import time\n",
    "\n",
    "# T\u1ef1 \u0111\u1ed9ng t\u1ea1o 20 c\u00e2u h\u1ecfi d\u1ef1a tr\u00ean c\u00e1c entities c\u00f3 th\u1eadt trong \u0111\u1ed3 th\u1ecb (G)\n",
    "top_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]]\n",
    "if not top_nodes:\n",
    "    top_nodes = [\"AI\", \"Technology\", \"Company\"] # Fallback n\u1ebfu graph r\u1ed7ng\n",
    "\n",
    "templates = [\n",
    "    \"What entities are connected to {}?\",\n",
    "    \"What products are connected to {}?\",\n",
    "    \"Which companies are connected through partnerships with {}?\",\n",
    "    \"Which companies are connected through acquisitions related to {}?\",\n",
    "    \"Which people are connected to {}?\",\n",
    "    \"What competitors are mentioned for {}?\",\n",
    "    \"Which companies are connected to AI products of {}?\",\n",
    "    \"What headquarters or locations are mentioned for {}?\",\n",
    "    \"What investments are mentioned for {}?\",\n",
    "    \"What is connected to {}?\",\n",
    "    \"What multi-hop relation can be found around {}?\",\n",
    "    \"Which products are connected to {}?\",\n",
    "    \"What are the most important relationships for {} in the graph?\",\n",
    "    \"Describe the ecosystem around {}.\",\n",
    "    \"What role does {} play in this context?\",\n",
    "    \"Who does {} interact with?\",\n",
    "    \"What technologies are linked to {}?\",\n",
    "    \"How does {} influence others?\",\n",
    "    \"What is the main focus of {}?\",\n",
    "    \"Summarize the connections of {}.\"\n",
    "]\n",
    "\n",
    "benchmark_questions = []\n",
    "for i in range(20):\n",
    "    node = top_nodes[i % len(top_nodes)]\n",
    "    benchmark_questions.append(templates[i].format(node))\n",
    "\n",
    "rows = []\n",
    "\n",
    "for q in benchmark_questions:\n",
    "    t0 = time.time()\n",
    "    flat_ans, flat_ctx = flat_rag_answer(q)\n",
    "    flat_time = time.time() - t0\n",
    "\n",
    "    t0 = time.time()\n",
    "    graph_ans, graph_ctx = graphrag_answer(q)\n",
    "    graph_time = time.time() - t0\n",
    "\n",
    "    rows.append({\n",
    "        \"question\": q,\n",
    "        \"flat_answer\": flat_ans,\n",
    "        \"graph_answer\": graph_ans,\n",
    "        \"flat_time\": flat_time,\n",
    "        \"graph_time\": graph_time,\n",
    "        \"flat_context_chars\": len(flat_ctx),\n",
    "        \"graph_context_chars\": len(graph_ctx),\n",
    "        \"winner\": \"\",\n",
    "        \"note\": \"\"\n",
    "    })\n",
    "\n",
    "import os\n",
    "BASE_DIR = os.getcwd()\n",
    "benchmark_df = pd.DataFrame(rows)\n",
    "benchmark_df.to_csv(os.path.join(BASE_DIR, \"benchmark_flat_vs_graphrag.csv\"), index=False)\n",
    "benchmark_df.head()"
]

nb["cells"][21]["source"] = nb_source

with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patch questions successful!")
