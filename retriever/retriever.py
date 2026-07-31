import os
import pickle
import faiss
import numpy as np
import re
import logging
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

VIETNAMESE_STOPWORDS = set([
    "là", "và", "của", "có", "được", "trong", "cho", "với", "các", "để",
    "này", "đó", "từ", "một", "theo", "về", "đã", "khi", "sẽ", "không",
    "thì", "như", "hoặc", "hay", "cũng", "nếu", "đến", "bởi", "tại",
    "vào", "ra", "lên", "xuống", "những", "mà", "nhưng", "còn", "rồi",
    "nên", "do", "vì", "bị", "chỉ", "đều", "trên", "dưới", "giữa",
    "trước", "sau", "qua"
])

# Token hoá tiếng việt để sử dụng với BM25
def tokenize_vietnamese(text):
    # Tokenize tiếng Việt cơ bản: lowercase, loại bỏ ký tự đặc biệt, loại stopwords
    text = text.lower().strip()
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]
    return tokens

class HybridRetriever:

    # Khởi tạo HybridRetriever với đường dẫn đến cơ sở dữ liệu vector
    def __init__(self, vector_db_path):
        print ("Đang tải dữ liệu và mô hình")

        # Tải metadata
        with open(os.path.join(vector_db_path, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)

        # tải FAISS index
        self.faiss_index = faiss.read_index(os.path.join(vector_db_path, "faiss_index.bin"))

        # tải mô hình embeddings (tối ưu VRAM: load lên CPU nếu CUDA khả dụng cho model lớn)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_device = 'cpu' if device == 'cuda' else device
        self.embeddings_model = SentenceTransformer('BAAI/bge-m3', device=embed_device)

        # chuẩn hóa nội dung văn bản để sử dụng với BM25, có sử dụng cache
        bm25_path = os.path.join(vector_db_path, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            print("Tải BM25 index từ cache...")
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            print("Xây dựng BM25 index mới...")
            tokenized_corpus = [tokenize_vietnamese(chunk["text"]) for chunk in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25, f)

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
        tokenized_query = tokenize_vietnamese(query)

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
    def hybrid_search(self, query, top_k=5, k_rrf = 60, min_score=None, filters=None):

        # Kết quả từ tìm kiếm semantic và keyword
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        # log các kết quả tìm kiếm
        logger.info(f"Query: '{query}'")
        logger.info("Semantic top 3: %s", [(r['chunk']['chunk_id'], round(float(r['score']), 4)) for r in semantic_results[:3]])
        logger.info("BM25 top 3: %s", [(r['chunk']['chunk_id'], round(float(r['score']), 4)) for r in keyword_results[:3]])

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
            # trọng số cho BM25 = 2.0 để ưu tiên các từ khóa chính xác (VD: "Sinh viên Xuất sắc", "Điều 16")
            rrf_scores[chunk_id]["rrf_score"] += 2.0 / (k_rrf + item["rank"])

        # Sắp xếp các kết quả dựa trên điểm số RRF và lấy top_k kết quả
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        if not sorted_results:
            return []
            
        # Áp dụng filter metadata sau RRF
        if filters:
            filtered = []
            for r in sorted_results:
                match = True
                for key, value in filters.items():
                    if key in r["chunk"] and value.lower() not in str(r["chunk"].get(key, "")).lower():
                        match = False
                        break
                if match:
                    filtered.append(r)
            sorted_results = filtered
            
        # Lọc theo ngưỡng điểm RRF tuyệt đối
        if min_score is not None:
            sorted_results = [r for r in sorted_results if r["rrf_score"] >= min_score]
            
        # Lọc theo tỷ lệ so với điểm cao nhất (adaptive threshold) để chống ảo giác
        if sorted_results:
            max_score = sorted_results[0]["rrf_score"]
            threshold = max_score * 0.3  # Giữ kết quả có điểm ≥ 30% so với top 1
            sorted_results = [r for r in sorted_results if r["rrf_score"] >= threshold]
            
        logger.info("RRF final top %d: %s", top_k, [(r['chunk']['chunk_id'], round(float(r['rrf_score']), 4)) for r in sorted_results[:top_k]])
            
        return sorted_results[:top_k]
