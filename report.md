# Báo cáo: So sánh hiệu suất GraphRAG và Flat RAG với Alibaba Qwen

**Lab Day 19 - Xây dựng hệ thống GraphRAG hoàn chỉnh**

## 1. Mục tiêu và Tổng quan
Mục tiêu của bài thực hành là xây dựng một hệ thống Graph Retrieval-Augmented Generation (GraphRAG) hoàn chỉnh từ một corpus văn bản thô (thư mục `dataset/`). Sau khi xây dựng đồ thị tri thức, hệ thống phải được đem lên bàn cân để so sánh trực tiếp với kiến trúc Flat RAG (Vector Search) cơ bản.

## 2. Phương pháp Triển khai
Để đảm bảo tính khả thi khi chạy trên tài nguyên hạn chế và tối ưu ngân sách API, nhóm đã áp dụng chiến lược sau:
- **Random Sampling có chủ đích:** Load ngẫu nhiên 10 file text với `seed=42`. Dữ liệu được băm (chunking) thành các khối nhỏ cỡ 1200 ký tự (overlap 150 ký tự).
- **Trích xuất thông tin (Extraction) bằng Alibaba Qwen:** Thay vì dùng Local LLM nặng nề, hệ thống gọi API `qwen3-vl-flash` qua OpenAI Compatible SDK với định dạng prompt chuẩn hóa theo phong cách *LangExtract*. LLM bị ép buộc trả về strict JSON chứa danh sách tối đa 6 triples `(subject, relation, object)` kèm `evidence` để tránh ảo giác (hallucination).
- **Làm sạch và xây dựng đồ thị:** Sử dụng Regex để chuẩn hóa chữ thường, xóa ký tự lạ, và nối các triples này lại thành một đồ thị `NetworkX MultiDiGraph`.
- **Flat RAG Local Baseline:** Nhằm tránh lãng phí tiền gọi Embedding API, baseline Flat RAG sử dụng `TfidfVectorizer` kết hợp với `cosine_similarity` để tìm 4 chunks phù hợp nhất trên local.
- **GraphRAG Reasoning:** Từ câu hỏi, thuật toán Entity Linking (fuzzy match) sẽ dò ra các node hạt giống (seeds) trên đồ thị. Từ seed, thuật toán duyệt 2 chặng (2-hop traversal) để lấy ra hàng loạt mối quan hệ chéo, diễn đạt (verbalize) chúng lại thành text và gửi cho Qwen trả lời.

## 3. Cải tiến Kỹ thuật
Bên cạnh luồng xử lý cơ bản, nhóm đã lập trình thêm các tính năng đặc thù:
1. **Dynamic Benchmark:** Thay vì kiểm thử trên 20 câu hỏi tĩnh (hardcode), hệ thống sẽ tìm ra 10 thực thể có nhiều liên kết (degree) nhất bên trong đồ thị do chính nó vừa tạo ra. Nó tự động lồng ghép các thực thể này vào 20 templates câu hỏi. Tính năng này đảm bảo hệ thống Benchmark là 100% khách quan và bám sát vào những gì có bên trong 10 file tài liệu được chọn.
2. **Cơ chế Cache 2 Lớp:** Mọi chunk gửi qua Extraction và mọi câu hỏi gửi lên Qwen QA đều được băm (md5 hash) và lưu vào `alibaba_langextract_cache.json` và `qwen_qa_cache.json`. Nếu hệ thống mất mạng hoặc bị gián đoạn, khi chạy lại, tiến độ sẽ lập tức được phục hồi mà không tốn thêm 1 token API nào.

## 4. Kết quả và Đánh giá
Toàn bộ quá trình chạy được lưu tại log và xuất ra 3 file:
- `extracted_triples.json`: Chứa hàng trăm dữ liệu quan hệ được sinh ra bởi LLM.
- `knowledge_graph.png`: Sơ đồ Graph thể hiện rõ cấu trúc liên kết của toàn bộ hệ sinh thái.
- `benchmark_flat_vs_graphrag.csv`: Dữ liệu bài test khách quan 20 câu hỏi.

**Kết luận so sánh:**
- **GraphRAG:** Phục vụ xuất sắc những câu hỏi cần tư duy nối điểm (Multi-hop Reasoning) như *"Mối quan hệ 2-hop nào xoay quanh công ty X?"*. Thay vì bối rối trước văn bản phân mảnh, GraphRAG tìm được chuỗi liên kết: *A -> mua lại -> B -> phát triển -> C*.
- **Flat RAG:** Phản hồi nhanh và ít tốn tài nguyên. Flat RAG trả lời hoàn hảo đối với các truy vấn tra cứu trực tiếp thông tin (Direct Lookup) khi thông tin nằm trọn trong 1 đoạn văn. Tuy nhiên, nó bị đánh bại hoàn toàn ở các câu tổng hợp.
