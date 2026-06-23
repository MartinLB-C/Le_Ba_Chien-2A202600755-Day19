# Lab Day 19 - Building a GraphRAG System (Alibaba Qwen API)

Dự án này triển khai một 파peline GraphRAG hoàn chỉnh từ tập dữ liệu văn bản thô và so sánh hiệu suất trực tiếp với Flat RAG (Vector Search) sử dụng **Alibaba DashScope Qwen API**.

## Các tính năng chính
- **Entity & Relation Extraction:** Trích xuất các bộ ba `(subject, relation, object)` sử dụng cơ chế LangExtract-style source-grounded prompt kết hợp với `qwen3-vl-flash`.
- **Knowledge Graph Construction:** Làm sạch, chuẩn hóa thực thể và xây dựng đồ thị mạng lưới đa hướng (MultiDiGraph) bằng thư viện `NetworkX`.
- **GraphRAG vs Flat RAG:** 
  - **GraphRAG:** 2-hop traversal, entity linking.
  - **Flat RAG:** Baseline với TF-IDF local index nhằm tiết kiệm chi phí embedding API.
- **Dynamic Benchmark:** Tự động phân tích top 10 thực thể lớn nhất trong đồ thị để sinh ra bộ test 20 câu hỏi độc nhất dựa trên dữ liệu thật.
- **Auto-Caching:** Tự động lưu cache cho cả quá trình Extraction và hỏi đáp (QA) nhằm tối ưu 100% quota API khi có sự cố.

## Hướng dẫn cài đặt và chạy thử nghiệm

**Bước 1: Chuẩn bị môi trường ảo**
```bash
python -m venv venv
.\venv\Scripts\activate   # (Windows)
# hoặc source venv/bin/activate (Mac/Linux)
```

**Bước 2: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

**Bước 3: Cấu hình API**
Sửa file `.env.example` thành `.env` và điền API Key DashScope của bạn:
```env
DASHSCOPE_API_KEY=your_alibaba_key
ALIBABA_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
ALIBABA_MODEL=qwen3-vl-flash
```

**Bước 4: Chạy toàn bộ Pipeline**
Xóa các file cache cũ (nếu có) trước khi chạy lần đầu:
```bash
Remove-Item alibaba_langextract_cache.json, qwen_qa_cache.json, extracted_triples.json
```
Sau đó khởi chạy tự động:
```bash
python run_pipeline.py
```
Hoặc bạn có thể dùng lệnh `jupyter notebook` và chạy tuần tự file `graphrag_alibaba.ipynb`.

## Kết quả xuất ra
- `extracted_triples.json`: Toàn bộ các bộ ba quan hệ trích xuất được.
- `knowledge_graph.png`: Đồ thị trực quan các mối quan hệ (nodes và edges).
- `benchmark_flat_vs_graphrag.csv`: Bảng điểm so sánh tốc độ, câu trả lời, và lượng ngữ cảnh nạp vào của 2 hệ thống.