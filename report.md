# Báo cáo: So sánh hiệu suất GraphRAG và Flat RAG với Alibaba Qwen

**Ngày thực hiện:** Hôm nay

## Mục tiêu
Lab Day 19 yêu cầu xây dựng một hệ thống GraphRAG hoàn chỉnh từ corpus text thô và so sánh hiệu suất, khả năng kết nối tri thức với Flat RAG (Vector Search) cơ bản.

## Phương pháp triển khai
Nhóm đã chọn ngẫu nhiên 10 file từ corpus với seed cố định (`seed=42`) để đảm bảo khả năng tái lập và dễ dàng kiểm thử. Dữ liệu sau đó được chia thành các chunk nhỏ (kích thước 1200 ký tự, overlap 150 ký tự). 

Thay vì xử lý toàn bộ corpus lớn (vốn tốn kém và đòi hỏi thời gian dài), hệ thống sử dụng phương pháp **LangExtract-style extraction** thông qua **Alibaba DashScope Qwen API** (model `qwen3-vl-flash`) để trích xuất các thông tin dưới dạng `(subject, relation, object)` triples từ các chunk được chọn.

Các bước chính:
1. **Extraction:** Prompt LLM trích xuất JSON schema chặt chẽ, giới hạn 6 triples mỗi chunk để tránh ảo giác.
2. **Deduplication & Graph Build:** Các triples sau đó được làm sạch bằng Regex, chuẩn hóa text và đưa vào `NetworkX MultiDiGraph`.
3. **GraphRAG Reasoning:** Hệ thống thực hiện entity linking, duyệt đồ thị 2-hop (2-hop traversal), sinh các câu fact dựa trên text và đưa context vào cho Qwen để trả lời các câu hỏi phức tạp.
4. **Flat RAG Baseline:** Dùng mô hình TF-IDF local (giúp giảm tải chi phí sử dụng Embedding API) để tìm kiếm top 4 chunks liên quan nhất và nạp vào Qwen.

## Kết quả
Toàn bộ quá trình chạy được theo dõi trong Jupyter Notebook `graphrag_alibaba.ipynb` và file log/kết xuất:
- `extracted_triples.json`: Chứa dữ liệu các node và cạnh được LLM trích xuất.
- `knowledge_graph.png`: Đồ thị trực quan các mối quan hệ (nodes và edges).
- `benchmark_flat_vs_graphrag.csv`: File CSV chứa 20 câu trả lời từ cả hai phương pháp cùng độ dài context và thời gian truy vấn.

## Kết luận
- **GraphRAG** thể hiện sức mạnh vượt trội ở các truy vấn nhiều chặng (multi-hop reasoning), giúp tổng hợp thông tin về quan hệ giữa các thực thể mà không một chunk đơn lẻ nào chứa đủ thông tin.
- **Flat RAG** nhanh và ổn định hơn ở các câu hỏi tra cứu trực tiếp một sự kiện nằm gọn trong một cụm text, nhưng hoàn toàn mất phương hướng nếu câu hỏi trải dài trên nhiều file.
