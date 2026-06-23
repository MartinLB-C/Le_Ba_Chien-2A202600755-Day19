import json

# --- Patch run_pipeline.py ---
pipeline_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\run_pipeline.py"
with open(pipeline_file, "r", encoding="utf-8") as f:
    code = f.read()

old_env_code = 'ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")\nALIBABA_MODEL = os.getenv("ALIBABA_MODEL", "qwen3-vl-flash")'
new_env_code = '''ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL")
ALIBABA_MODEL = os.getenv("ALIBABA_MODEL")

if not ALIBABA_BASE_URL or not ALIBABA_MODEL:
    print("Vui lòng cấu hình ALIBABA_BASE_URL và ALIBABA_MODEL trong file .env!")
'''
code = code.replace(old_env_code, new_env_code)

with open(pipeline_file, "w", encoding="utf-8") as f:
    f.write(code)

# --- Patch graphrag_alibaba.ipynb ---
nb_file = r"d:\VIn_GD2\day19\Le_Ba_Chien-2A202600755-Day19\graphrag_alibaba.ipynb"
with open(nb_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

old_nb_source = [
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

new_nb_source = [
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
    "ALIBABA_BASE_URL = os.getenv(\"ALIBABA_BASE_URL\")\n",
    "ALIBABA_MODEL = os.getenv(\"ALIBABA_MODEL\")\n",
    "if not ALIBABA_BASE_URL or not ALIBABA_MODEL:\n",
    "    print(\"Vui l\u00f2ng c\u1ea5u h\u00ecnh ALIBABA_BASE_URL v\u00e0 ALIBABA_MODEL trong file .env!\")\n",
    "\n",
    "MAX_TRIPLES_PER_CHUNK = 6\n",
    "MAX_CHARS_PER_CHUNK_FOR_API = 4000\n",
    "MAX_EXTRACTION_CALLS = 30"
]

nb["cells"][3]["source"] = new_nb_source

with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patch strict ENV successful!")
