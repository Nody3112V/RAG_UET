import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:

    # Khởi tạo HybridRetriever với đường dẫn đến cơ sở dữ liệu vector
    def __init__(self, vector_db_path):
        print ("Đang tải dữ liệu và mô hình")

        # Tải metadata
        with open(os.path.join(vector_db_path, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)

        # tải FAISS index
        self.faiss_index = faiss.read_index(os.path.join(vector_db_path, "faiss_index.bin"))

        # tải mô hình embeddings
        self.embeddings_model = SentenceTransformer('BAAI/bge-m3', model_kwargs={"use_safetensors": True})

        # chuẩn hóa nội dung văn bản để sử dụng với BM25
        tokenized_corpus = [chunk["text"].lower().strip().split() for chunk in self.metadata]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Đã tải xong dữ liệu và mô hình")

    # Hàm tìm kiếm cosine similarity dựa trên FAISS index
    def semantic_search(self, query, top_k=5):

        # Mã hóa truy vấn thành vector embeddings
        query_vector = self.embeddings_model.encode([query], normalize_embeddings=True)

        # Tìm kiếm trong FAISS index để lấy các chỉ số của các chunks gần nhất
        distances, indices = self.faiss_index.search(query_vector, top_k)

        # Danh sách các kết quả tìm kiếm dựa trên chỉ số
        results = []

        # Lặp qua các chỉ số và khoảng cách để tạo danh sách kết quả
        # kết quả kèm điểm số và xếp hạng các chunk, truyền toàn bộ top_k vào để RRF xử lý
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.metadata[idx],
                    "score": distances[0][i],
                    "rank": len(results) + 1
                })

        return results

    # Hàm tìm kiếm dựa trên từ khóa sử dụng BM25
    def keyword_search(self, query, top_k=5):
        # Token hóa truy vấn 
        tokenized_query = query.lower().strip().split()

        # Tính điểm số BM25 cho từng chunk dựa trên truy vấn
        scores = self.bm25.get_scores(tokenized_query)

        # Lấy các chỉ số của các chunk có điểm số cao nhất
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Tạo danh sách kết quả dựa trên các chỉ số và điểm số
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "chunk": self.metadata[idx],
                    "score": scores[idx],
                    "rank": i + 1
                })
    
        return results

    # Hàm tìm kiếm kết hợp giữa semantic search và keyword search
    def hybrid_search(self, query, top_k=5, k_rrf = 60):

        # Kết quả từ tìm kiếm semantic và keyword
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # Dictionary để lưu trữ điểm số RRF cho từng chunk
        rrf_scores = {}

        # Tính điểm số RRF cho từng chunk dựa trên kết quả từ cả hai phương pháp tìm kiếm
        for item in semantic_results:
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": item["chunk"], "rrf_score": 0.0, "semantic_score": item["score"]}
            rrf_scores[chunk_id]["rrf_score"] += 1.0 / (k_rrf + item["rank"])

        for item in keyword_results:
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": item["chunk"], "rrf_score": 0.0, "semantic_score": 0.0}
            # Tăng trọng số cho BM25 lên 2.0 để ưu tiên các từ khóa chính xác (VD: "Sinh viên Xuất sắc", "Điều 16")
            rrf_scores[chunk_id]["rrf_score"] += 2.0 / (k_rrf + item["rank"])

        # Sắp xếp các kết quả dựa trên điểm số RRF và lấy top_k kết quả
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        if not sorted_results:
            return []
            
        return sorted_results[:top_k]

# --- Chạy thử nghiệm module ---
if __name__ == "__main__":
    # Đảm bảo trỏ đúng thư mục vector_db bạn đã tạo ở bước Ingestion
    vector_db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vector_db")
    
    try:
        retriever = HybridRetriever(vector_db_dir)
        
        # Thử nghiệm với một câu hỏi
        cau_hoi = "Học kỳ phụ có bao nhiêu tuần học?"
        print(f"\nCâu hỏi: {cau_hoi}")
        
        ket_qua = retriever.hybrid_search(cau_hoi, top_k=3)
        
        for i, kq in enumerate(ket_qua, 1):
            print(f"\n--- Top {i} (Điểm RRF: {kq['rrf_score']:.4f}) ---")
            chuong_info = kq['chunk'].get('chuong', '')
            chuong_str = f" | {chuong_info}" if chuong_info else ""
            print(f"Nguồn: {kq['chunk']['source']}{chuong_str} | Article: {kq['chunk']['article_id']}")
            print(f"Nội dung trích xuất: {kq['chunk']['text'][:200]}...")
            
    except Exception as e:
        print(f"Có lỗi xảy ra, hãy kiểm tra lại đường dẫn database: {e}")