# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Thế Hưng  
**Mã số học viên:** 2A202602381  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9800 | 0.8947 | -0.0853 |
| Answer Relevancy | 0.4914 | 0.5788 | +0.0873 |
| Context Precision | 0.4103 | 0.5444 | +0.1342 |
| Context Recall | 0.3571 | 0.5882 | +0.2311 |

> **Nhận xét tổng quan:**  
> Hệ thống **Production RAG** ghi nhận bước nhảy vọt về khả năng truy xuất thông tin liên quan:
> - **Context Recall** tăng mạnh từ **0.3571 lên 0.5882 (+0.2311, tăng ~65%)**: Nhờ kết hợp Hybrid Search (BM25 tách từ tiếng Việt + Dense vector) và làm giàu ngữ cảnh Contextual Prepending, hệ thống bao phủ thông tin từ tài liệu gốc tốt hơn hẳn so với naive dense search.
> - **Context Precision** tăng từ **0.4103 lên 0.5444 (+0.1342)**: Nhờ có bước Reranking (FlashRank) tái sắp xếp các candidate, lọc nhiễu và đẩy các chunk liên quan nhất lên top-3 context.
> - **Answer Relevancy** tăng từ **0.4914 lên 0.5788 (+0.0873)**: Câu trả lời bám sát trọng tâm câu hỏi hơn nhờ context đầu vào cô đọng và chất lượng.
> - **Faithfulness (0.8947)**: Đạt mức cao an toàn (>0.75 chuẩn rubric), đảm bảo câu trả lời trung thực với context được cung cấp.

---

## Bottom-5 Failures

### #1
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (Score: 0.0)
- **Error Tree:** Output sai (trả lời 'Không tìm thấy') → Context sai/thiếu (Retriever không kéo đủ chunk chứa quy chế lương và chính sách phân loại dữ liệu bảo mật đồng thời) → Query OK nhưng cần multi-hop reasoning.
- **Root cause:** Câu hỏi đòi hỏi thông tin bắc cầu (multi-hop) giữa 2 tài liệu: `ky_luong.md` (nói về thông tin lương là dữ liệu bí mật) và `phan_loai_du_lieu.md` (định nghĩa cấp độ 3 - Bí mật). Retriever đơn thuần chỉ lấy top 3 reranked chunks nên không gom đủ cả 2 nguồn, khiến context không đủ điều kiện cho LLM trả lời, kích hoạt cơ chế từ chối trả lời ("Không tìm thấy").
- **Suggested fix:** Áp dụng Multi-Query Expansion hoặc Sub-question Decomposition để truy vấn độc lập từng thực thể ("thông tin lương" và "cấp độ phân loại dữ liệu") rồi gộp kết quả trước khi rerank.

### #2
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Got:** Không tìm thấy.
- **Worst metric:** Context Recall (Score: 0.0)
- **Error Tree:** Output sai → Context sai (thiếu chunk từ file `nghi_phep_dac_biet.md`) → Query BM25/Dense bị phân tán bởi các chunk phép năm và nghỉ ốm có tần suất từ khóa cao hơn.
- **Root cause:** Từ khóa "kết hôn" chỉ xuất hiện trong một đoạn ngắn của chính sách nghỉ phép đặc biệt. Trong quá trình hybrid search và reranking, các chunk nói về "nghỉ phép" nói chung có độ tương đồng ngữ nghĩa cao lấn át chunk đặc biệt có chứa từ khóa "kết hôn", dẫn đến việc chunk mục tiêu bị rơi khỏi top-3 context cuối cùng.
- **Suggested fix:** Tăng trọng số cho BM25 đối với các keyword mang tính định danh đặc thù ("kết hôn"), hoặc tăng `RERANK_TOP_K` từ 3 lên 5 để mở rộng context window cho generator.

### #3
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Không tìm thấy.
- **Worst metric:** Context Recall (Score: 0.0)
- **Error Tree:** Output sai → Context bị mâu thuẫn giữa 2 phiên bản (v2023 và v2024) → Prompt hướng dẫn "Nếu không có hoặc không chắc chắn → nói 'Không tìm thấy'".
- **Root cause:** Bộ dữ liệu có cả 2 tài liệu: `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày). Khi cả hai chunk đều được kéo vào, nếu LLM thấy xung đột phiên bản mà prompt không có chỉ thị ưu tiên phiên bản mới nhất, hệ thống từ chối đưa ra kết luận hoặc trượt recall.
- **Suggested fix:** Bổ sung metadata filtering theo `version` / `date` hoặc thêm chỉ dẫn trong System Prompt: "Ưu tiên chính sách có phiên bản mới nhất (v2024 > v2023)".

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Không tìm thấy.
- **Worst metric:** Context Recall (Score: 0.0)
- **Error Tree:** Output sai → Context thiếu mảnh ghép điều kiện kỹ thuật CNTT từ file mua sắm.
- **Root cause:** Câu hỏi có 2 vế điều kiện: thẩm quyền tài chính (30 triệu -> Giám đốc) và thủ tục kỹ thuật CNTT (xác nhận cấu hình từ phòng CNTT). Hai thông tin này nằm ở hai phần khác nhau của tài liệu `mua_sam.md`. Child chunk 256 ký tự quá nhỏ để chứa trọn vẹn cả 2 quy định nếu chúng nằm cách xa nhau.
- **Suggested fix:** Tận dụng triệt để Parent Chunk Retrieval (khi một Child chunk được retrieve, nạp luôn Parent chunk tương ứng vào context) để cung cấp toàn cảnh văn bản.

### #5
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Không tìm thấy.
- **Worst metric:** Context Recall (Score: 0.0)
- **Error Tree:** Output sai → Context thiếu quy định xử lý quá hạn tạm ứng → LLM không tự tính toán suy diễn khi thiếu context.
- **Root cause:** Tài liệu `tam_ung.md` có chứa quy định thời hạn 15 ngày và mức phạt 2%/tháng, nhưng câu hỏi yêu cầu áp dụng tính toán tình huống thực tế (sau 20 ngày -> quá hạn 5 ngày). Do prompt generator yêu cầu "Chỉ trả lời dựa trên context, nếu không có nói Không tìm thấy", LLM e ngại suy diễn tính toán số học khi context không nêu đích danh số tiền phạt của 5 ngày quá hạn.
- **Suggested fix:** Cải thiện prompt template bổ sung khả năng "Chain-of-Thought (CoT)" cho các câu hỏi tình huống tính toán số học, cho phép mô hình suy luận logic dựa trên các con số định lượng có trong context.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Đúng một phần. Output trả về: *"Nhân viên có 9 năm thâm niên được 18 ngày phép năm. Thông tin về lương không được đề cập trong tài liệu."* → Đạt yêu cầu về số ngày phép nhưng bị thiếu hoàn toàn vế thông tin dải lương Senior.
2. **Context đúng?** → Sai/Thiếu. Context Precision bị giảm và Context Recall chỉ đạt một phần. Retriever chỉ kéo được chunk về thâm niên từ `nghi_phep_nam_v2024.md`, hoàn toàn bỏ sót chunk từ `bang_luong_2024.md` (nơi quy định cấp bậc Senior P3-P4 có mức lương 20-35 triệu).
3. **Query rewrite OK?** → Chưa tối ưu. Query là câu hỏi ghép gồm 2 thực thể ngữ nghĩa xa nhau: "ngày phép năm thâm niên" (chính sách nhân sự) và "dải lương Senior" (chế độ đãi ngộ tài chính). Một vector duy nhất không thể tối ưu khoảng cách tới cả 2 miền dữ liệu riêng biệt này.
4. **Fix ở bước:** Bước **Query Pre-processing (Query Decomposition)**. Tách câu hỏi ghép thành 2 sub-queries độc lập:
   - Sub-query 1: *"Nhân viên có 9 năm thâm niên được cộng bao nhiêu ngày phép năm?"*
   - Sub-query 2: *"Dải lương của nhân viên cấp Senior là bao nhiêu?"*  
   Sau đó thực hiện multi-retrieval và hợp nhất context trước khi gửi vào LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Parent Document Retrieval:** Khi Child chunk match, tự động nạp Parent chunk (2048 chars) chứa trọn vẹn văn cảnh xung quanh, giải quyết triệt để vấn đề mất thông tin liên quan giữa các điều khoản.
- **Query Decomposition & Expansion:** Bổ sung module sinh truy vấn con tự động cho các câu hỏi ghép/phức hợp.
- **Temporal / Version Filtering:** Bổ sung metadata filtering theo `version: v2024` để tự động loại bỏ các văn bản cũ đã hết hiệu lực, tránh gây nhiễu cho generator.
