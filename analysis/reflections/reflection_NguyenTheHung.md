# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Thế Hưng  
**Mã số học viên:** 2A202602381  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code đã triển khai trong bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Sử dụng cosine similarity giữa các câu liên tiếp với ngưỡng threshold 0.85 (dựa trên sentence embedding). Khi khoảng cách ngữ nghĩa vượt ngưỡng, hệ thống tạo ranh giới chunk mới, giúp gom các câu cùng một mạch ý nghĩa thay vì cắt ngang tùy tiện theo độ dài token cố định. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Tách tài liệu thành các Parent chunk (2048 chars) chứa ngữ cảnh tổng quan và các Child chunk (256 chars) chứa chi tiết. Child chunk lưu `parent_id` liên kết, giải quyết xung đột kinh điển giữa retrieval precision (cần chunk nhỏ) và synthesis context (cần context rộng). |
| **BM25 + Dense fusion** | M2 | `reciprocal_rank_fusion()` | RRF hợp nhất danh sách xếp hạng từ BM25 (lexical matching sử dụng Underthesea tách từ tiếng Việt) và Dense Search (vector cosine trong Qdrant). Công thức $RRF(d) = \sum \frac{1}{k + rank_i(d)}$ với $k=60$ giúp loại bỏ chênh lệch về thang điểm (scale variance), tận dụng tối đa thế mạnh bắt từ khóa chính xác (mã quy định, phiên bản v2023/v2024) và bắt ngữ nghĩa đồng nghĩa. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | Bi-encoder trong khâu retrieval chỉ tính toán vector độc lập cho câu hỏi và tài liệu (nhanh nhưng mất tương tác chéo). Cross-encoder nhận cặp `(query, document)` và tính all-to-all cross-attention giữa mọi token, chấm điểm mức độ liên quan sâu sắc. Rerank từ top 20 candidate xuống top 3 context chất lượng cao nhất giúp tăng vọt `context_precision`. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | Đánh giá toàn diện 4 trụ cột RAG: Faithfulness (đo lường tỷ lệ khẳng định trong câu trả lời có căn cứ từ context - chống ảo giác), Answer Relevancy (đo độ bám sát câu hỏi), Context Precision (tỷ lệ chunk liên quan nằm ở thứ hạng đầu), và Context Recall (mức độ bao phủ thông tin từ ground truth). |
| **Contextual Enrichment** | M5 | `contextual_prepend()` / `_enrich_single_call()` | Trước khi đưa vào vector index, chunk nhỏ được bổ sung thêm 1 câu tóm tắt vị trí/ngữ cảnh tài liệu (`Contextual Prepend`), bộ câu hỏi giả định (`HyQA`) và metadata tự động (topic, entities). Kỹ thuật này khắc phục triệt để hiện tượng chunk mất ngữ cảnh khi đứng độc lập. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. `huggingface_hub.utils._errors.HfHubHTTPError / ConnectTimeoutError: Connection to huggingface.co timed out`: Khi tải mô hình `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3` (mỗi mô hình nặng hơn 2.2GB), mạng kết nối từ môi trường Windows trong nước không có `HF_TOKEN` bị timeout hoặc ngắt kết nối giữa chừng, để lại file lock `.locks` gây treo tiến trình.
  2. `google.api_core.exceptions.ResourceExhausted: 429 You have exhausted your daily quota of requests for this model. The model 'gemini-3.8-flash' has a quota limit of 20 requests per day.`: Khi sử dụng Gemini API thông qua endpoint OpenAI-compatible, model preview bị chạm trần quota hàng ngày rất nhanh, đồng thời Ragas gọi async với 16 workers đồng thời dẫn tới nghẽn rate limit.
  3. `qdrant_client.http.exceptions.UnexpectedResponse: Vector dimension error: expected 1024, got 384`: Khi chuyển đổi mô hình embedding sang `all-MiniLM-L6-v2`, kích thước vector config ban đầu bị gán cứng là 1024.

- **Nguyên nhân gốc rễ & Cách debug:**
  1. *Khắc phục mô hình Embedding & Reranking:* Dọn dẹp cache lock file; đối với embeddings sử dụng mô hình tối ưu `all-MiniLM-L6-v2` (đã được cache sẵn hoàn chỉnh trong máy, kích thước nhẹ ~90MB, nạp `local_files_only=True` chỉ mất 0.3s). Đối với Reranking, tích hợp fallback thông minh sang `FlashrankReranker` (`ms-marco-TinyBERT-L-2-v2`, kích thước chỉ 3.2MB, độ trễ suy luận <5ms), đáp ứng hoàn hảo tiêu chí nhanh và chính xác. Đồng thời cấu hình `DenseSearch` tự động trích xuất vector dimension (`encoder.get_sentence_embedding_dimension()`) khi tạo collection trên Qdrant.
  2. *Khắc phục Rate Limit & Concurrency:* Chuyển đổi sang model `gemini-3.5-flash-lite` với quota dồi dào và tốc độ xử lý nhanh; xây dựng cơ chế persistent disk cache (`reports/enrichment_cache.json`) cho Module M5 Enrichment để không gọi trùng lặp API khi chạy lại pipeline. Trong Ragas evaluation, bổ sung cấu hình `RunConfig(max_workers=3, max_retries=5)` để kiểm soát lưu lượng request, loại bỏ hoàn toàn lỗi 429.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cần hiểu sâu hơn về kiến trúc bất đồng bộ (async event loop) của RAGAS khi tương tác với các LLM provider có rate limits thấp. Đã khắc phục bằng cách đọc tài liệu chính thức của Ragas về `RunConfig` và `TokenUsageParser`.
  - Hiểu rõ sự khác biệt giữa Lexical Search (BM25) khi xử lý tiếng Việt: tiếng Việt là ngôn ngữ đơn lập, ranh giới từ gồm nhiều tiếng (từ ghép) nên bắt buộc phải qua phân tích hình thái từ vị (word segmentation bằng `underthesea`) trước khi tokenize cho BM25.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy định Doanh nghiệp (Enterprise Legal & Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng mô hình Naive RAG cơ bản: cắt văn bản theo số lượng ký tự cố định (fixed-size chunking 500 ký tự), dùng OpenAI text-embedding-3-small và similarity search top 5 trên Qdrant.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context fragmentation:* Các điều khoản pháp lý thường có cấu trúc phân cấp (Chương -> Điều -> Khoản -> Điểm). Cắt cố định làm mất tiêu đề điều khoản ở các chunk sau.
  - *Version collision:* Khi có nhiều phiên bản quy chế (ví dụ Quy chế 2023 và Quy chế sửa đổi 2024), Dense search đơn thuần thường kéo nhầm các quy định đã hết hiệu lực do độ tương đồng ngữ nghĩa quá cao.
  - *Latency & Precision:* Lấy trực tiếp top 5 gửi vào context gây loãng thông tin, đôi khi xuất hiện ảo giác (hallucination) ở những câu hỏi về mốc thời gian hoặc số liệu cụ thể.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Structure-Aware Chunking** làm nòng cốt kết hợp **Hierarchical Chunking**. Nhận diện các thẻ Markdown / Header điều khoản (`Điều 1`, `Khoản 2...`) để giữ nguyên cấu trúc logic. Parent chunk lưu toàn bộ Điều luật, Child chunk lưu từng Khoản cụ thể kèm `parent_id`.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense) kết hợp RRF ($k=60$)**. BM25 với bộ tách từ tiếng Việt giúp bắt chính xác các số hiệu văn bản, mã điều khoản và năm ban hành; Dense search bắt các câu hỏi diễn đạt tự nhiên của người dùng.
3. **Reranking:** Tích hợp **Cross-Encoder / FlashRank** để tái xếp hạng top 20 candidate xuống top 3 trước khi đưa vào LLM Context. Giúp loại bỏ các văn bản cũ hoặc không khớp chính xác điều kiện áp dụng.
4. **Enrichment:** Triển khai **Contextual Prepending** tự động chèn thông tin: `[Văn bản: Quy chế nội bộ | Hiệu lực: 2024 | Trạng thái: Đang áp dụng]` vào đầu mỗi chunk trước khi index.
5. **Evaluation:** Thiết lập bộ benchmark tự động bằng **RAGAS 4 metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) chạy định kỳ trong CI/CD để giám sát chất lượng khi cập nhật văn bản mới.

#### 3. Timeline triển khai
- **Tuần 1 (Data Preprocessing & Indexing):**
  - Viết parser cấu trúc chuyên dụng cho văn bản quy phạm và tài liệu nội bộ.
  - Xây dựng pipeline Structure-aware chunking + Contextual enrichment.
  - Thiết lập Hybrid indexing trên Qdrant (BM25 + Dense vector).
- **Tuần 2 (Reranking, Serving & Evaluation):**
  - Tích hợp lớp FlashRank Reranker vào inference API.
  - Viết bộ test set 50 câu hỏi nghiệp vụ thực tế có ground truth.
  - Chạy đánh giá RAGAS, phân tích Diagnostic Tree để tinh chỉnh prompt hệ thống và hoàn thiện pipeline đưa lên môi trường staging.
