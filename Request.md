# Tìm hiểu và Xây dựng hệ thống RAG ứng dụng vào chatbot hỏi đáp 

Nhiệm vụ là tìm hiểu các kiến thức liên quan đến kỹ thuật RAG và thực hiện xây dựng một hệ thống chatbot hỏi đáp tự động dựa trên phương pháp **RAG (Retrieval-Augmented Generation)**.

---

## 🎯 Mục tiêu
Xây dựng trọn vẹn một ứng dụng RAG Chatbot từ đầu đến cuối (End-to-End RAG Application), bao gồm:
1. **Crawl/Scrape dữ liệu**
2. **Tiền xử lý (Ingestion/Embedding)**
3. **Thiết kế luồng truy hồi (Retrieval)**
4. **Sinh câu trả lời (Generation)** chính xác bằng LLM.
5. **Phát triển giao diện Chatbot UI**

---

## 🔒 Các Yêu cầu và Ràng buộc Cố định (Constraints)

Để đảm bảo tính công bằng và dễ đánh giá, **BẮT BUỘC** phải tuân thủ các cấu hình phần cứng/phần mềm sau:

### 1. Nguồn dữ liệu (Data Source)
*   **Target Website:** [Sổ tay Sinh viên UET](https://handbook.uet.vnu.edu.vn/)
*   Tự viết công cụ crawl toàn bộ nội dung web, bao gồm văn bản trên các trang con và tải về/trích xuất nội dung của tất cả các file đính kèm dạng `.pdf`, `.docx`, e.g. (nếu có) trên trang này.

### 2. Mô hình Embedding cố định
*   **Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
*   *Lưu ý:* Không sử dụng bất kỳ mô hình nhúng (Embedding) nào khác.

### 3. Mô hình Ngôn ngữ lớn (LLM) cố định
*   **Model:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
*   *Lưu ý:* Không sử dụng các mô hình khác để đảm bảo đều chạy trên cùng một giới hạn tài nguyên tính toán (Kaggle GPU).

### 4. Giao diện Chatbot (User Interface)
*   Xây dựng giao diện ứng dụng hoàn chỉnh để người dùng có thể gõ câu hỏi, nhận câu trả lời dạng chat. Giao diện cần trực quan, hỗ trợ hiển thị lịch sử trò chuyện (Chat History) và nguồn trích dẫn dữ liệu (Source Citations).
*   Công nghệ khuyến nghị: *Streamlit, Gradio, hoặc HTML/CSS/JS kết hợp FastAPI/Flask backend.*

---

## 🏗️ Kiến trúc các Module cần Phát triển

Hệ thống của đội thi nên được thiết kế theo cấu trúc mô-đun hóa rõ ràng:

```text
rag-pipeline/
├── crawler/                  # Module cào và tải dữ liệu web/pdf/docx
├── ingestion/                # Module phân tách đoạn (chunking) và lập chỉ mục Vector
├── retriever/                # Module tìm kiếm ngữ nghĩa và từ khóa (Hybrid Search/Rerank)
├── app.py                 # File chạy giao diện Chatbot chính
├── requirements.txt          # Danh sách thư viện cần cài đặt
└── README.md                 # Tài liệu hướng dẫn sử dụng hệ thống của đội thi
```

### Chi tiết các bước thực hiện:

### Bước 1: Crawl Dữ liệu (`crawler/`)
*   Viết script cào tự động duyệt qua cấu trúc cây thư mục (sitemap) của `https://handbook.uet.vnu.edu.vn/`.
*   Tải xuống các văn bản quy chế PDF/DOCX và chuyển đổi sang dạng văn bản thô (Text).

### Bước 2: Phân tách & Nhúng dữ liệu (`ingestion/`)
*   Quy chế UET có cấu trúc pháp lý dạng **Chương - Điều - Khoản**. Hãy thiết kế giải thuật phân đoạn (Chunking) thông minh để gom nhóm từng Điều luật thành một chunk thống nhất thay vì cắt chia độ dài cố định một cách ngẫu nhiên. Ví dụ:

```bash
{
  "chunk_id": "quy_che_dao_tao_k67.docx_Điều_16_22", 
  "article_id": "Điều_16", 
  "source": "quy_che_dao_tao_k67.docx", 
  "text": "Điều 16. Học kỳ | Mỗi năm học có hai học kỳ chính và một học kỳ phụ. Mỗi học kỳ chính có 15 tuần học; từ 3 dến 4 tuần thi và 1 tuần dự phòng. \nMỗi học kỳ phụ có ít nhất 5 tuần học và 1 tuần thi; được tổ chức trong thời gian giữa hai học kỳ chính."
}
...
```
*   Sử dụng mô hình `BAAI/bge-m3` nhúng toàn bộ các chunk văn bản thành Dense Vectors và lưu trữ vào cơ sở dữ liệu Vector (Vector DB như FAISS, Chroma, Qdrant hoặc lưu file nhị phân cục bộ).

### Bước 3: Thiết kế Bộ truy hồi (`retriever/`)
*   Xây dựng luồng tìm kiếm kết hợp (Hybrid Search) giữa:
    *   **Keyword Search:** Sử dụng BM25 để khớp chính xác các từ chuyên ngành (Ví dụ: tên môn học, mã điều lệ).
    *   **Semantic Search:** So khớp Cosine Similarity dựa trên vector nhúng của `bge-m3`.
*   Kết hợp thứ hạng bằng thuật toán **RRF (Reciprocal Rank Fusion)** hoặc các giải pháp lọc/rerank nâng cao.

### Bước 4: Tích hợp LLM sinh câu trả lời
*   Xây dựng Prompt System chuẩn hóa tiếng Việt, hướng dẫn `Qwen2.5-3B-Instruct` đọc hiểu ngữ cảnh và đưa ra câu trả lời ngắn gọn.
*   **Cơ chế chống ảo tưởng (Anti-Hallucination):** Nếu thông tin truy hồi không chứa câu trả lời, LLM bắt buộc phải trả về câu từ chối an toàn (Ví dụ: *"Tôi không có đủ dữ liệu để trả lời"*), tránh việc tự bịa ra quy chế học tập.

### Bước 5: Xây dựng Giao diện Chatbot (UI)
*   Thiết kế giao diện đẹp mắt, thân thiện.
*   Hỗ trợ lưu ngữ cảnh hội thoại (Multi-turn Conversation) để người dùng hỏi các câu tiếp nối (Ví dụ: *"Điều kiện nhận học bổng là gì?"* -> *"Mức học bổng loại xuất sắc là bao nhiêu?"*). Tức là model cần nhớ được context.
*   **Hiển thị nguồn trích dẫn:** Chatbot khi trả lời phải ghi rõ thông tin này được lấy từ đâu để người dùng đối chiếu.

---

## 🏆 Tiêu chí Đánh giá (Evaluation Criteria)

Đánh giá dựa trên 4 tiêu chí:

| Tiêu chí | Trọng số | Mô tả chi tiết |
| :--- | :---: | :--- |
| **Chất lượng Crawl & Phân đoạn dữ liệu** | **30%** | Mức độ đầy đủ của dữ liệu thu thập, giải thuật phân tách Chương - Điều luật giữ nguyên ngữ nghĩa. |
| **Độ chính xác RAG (Answer Quality)** | **20%** | Chatbot trả lời đúng trọng tâm câu hỏi, không bị ảo tưởng thông tin, nhận diện tốt các trường hợp phủ định hoặc không có dữ liệu. |
| **Tối ưu hóa Truy hồi & Tốc độ** | **40%** | Tốc độ tìm kiếm ngữ cảnh, cấu hình sử dụng tài nguyên GPU hiệu quả, áp dụng Hybrid Search/Rerank hợp lý. |
| **Trải nghiệm giao diện Chatbot UI** | **10%** | Giao diện trực quan đẹp mắt, khả năng hiển thị nguồn trích dẫn rõ ràng, xử lý hội thoại đa ngữ cảnh tốt. |