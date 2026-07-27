# 🎓 UET AI Chatbot - RAG Pipeline

Hệ thống Chatbot hỏi đáp tự động cho Sổ tay sinh viên UET dựa trên phương pháp **RAG (Retrieval-Augmented Generation)**.

## 🏗️ Kiến trúc Hệ thống

Hệ thống được thiết kế theo cấu trúc mô-đun hóa:
- **Crawler (`crawler/crawler.py`)**: Công cụ tự động thu thập toàn bộ dữ liệu từ `https://handbook.uet.vnu.edu.vn/`, tự động phát hiện và trích xuất nội dung từ các file `PDF`, `DOCX` bằng OCR (`pytesseract`, `pdfplumber`).
- **Ingestion (`ingestion/ingestion.py`)**: Phân tách văn bản thông minh theo định dạng (Chương - Điều - Khoản). Sử dụng mô hình `BAAI/bge-m3` để nhúng (embedding) và lưu trữ cục bộ dưới dạng vector với FAISS (`faiss-cpu`).
- **Retriever (`retriever/retriever.py`)**: Thiết kế bộ truy hồi thông minh sử dụng **Hybrid Search**, kết hợp Keyword Search (`BM25`) và Semantic Search (Cosine Similarity trên FAISS), sau đó gộp điểm và xếp hạng lại bằng công thức **RRF (Reciprocal Rank Fusion)**.
- **Generator (`generator/generator.py`)**: Sử dụng Mô hình Ngôn ngữ Lớn (LLM) `Qwen/Qwen2.5-3B-Instruct` để sinh ra câu trả lời tiếng Việt mượt mà. Hệ thống Prompt được tinh chỉnh nghiêm ngặt để **chống ảo tưởng thông tin (Anti-Hallucination)**.
- **Giao diện Chatbot (`app.py`)**: Xây dựng bằng Streamlit, hỗ trợ Multi-turn Conversation (Hội thoại đa lượt) và hiển thị trực quan các nguồn trích dẫn.

## ⚙️ Cài đặt Môi trường

Yêu cầu sử dụng Python 3.9 trở lên.

1. Clone hoặc tải mã nguồn về.
2. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
# Window
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
3. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```
4. **Cài đặt Poppler**:
Mô-đun Crawler sử dụng `pdf2image` để chuyển đổi PDF sang ảnh, yêu cầu bắt buộc phải cài đặt Poppler trên hệ thống.
- **Windows**: Tải phiên bản mới nhất tại [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/), giải nén và cấu hình biến môi trường `PATH` trỏ tới thư mục `bin` của poppler.
- **Linux/Ubuntu**: `sudo apt-get install poppler-utils`
- **Mac**: `brew install poppler`

5. **Cài đặt Tesseract OCR**:
Mô-đun Crawler sử dụng Tesseract để đọc các file PDF định dạng ảnh.
- **Windows**: Tải và cài đặt tại [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Sau đó cấu hình biến môi trường `PATH` trỏ tới thư mục cài đặt Tesseract.
- **Linux/Ubuntu**: `sudo apt-get install tesseract-ocr tesseract-ocr-vie`

## 🚀 Hướng dẫn Chạy

Thực hiện lần lượt các bước sau:

**Bước 1: Thu thập dữ liệu (Crawl)**
```bash
python crawler/crawler.py
```
*(Kết quả sẽ được lưu vào thư mục `data/raw/` dưới dạng file json và log)*

**Bước 2: Xử lý và Nhúng dữ liệu (Ingestion)**
```bash
python ingestion/ingestion.py
```
*(Mô hình bge-m3 sẽ chạy và vector sinh ra được lưu tại thư mục `data/vector_db/`)*

**Bước 3: Khởi động Chatbot (App)**
```bash
streamlit run app.py
```
*(Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`)*

## 💡 Lưu ý về LLM
**Khuyến nghị:** Do yêu cầu cấu hình phần cứng tương đối cao, bạn nên chạy mã nguồn này trên nền tảng **Kaggle** và bật bộ tăng tốc **GPU T4** để đạt tốc độ nhúng (embedding) và sinh văn bản (text-generation).

---

