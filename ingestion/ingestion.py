import json
import re
import os
import pickle
import faiss
import sys
from sentence_transformers import SentenceTransformer
from urllib.parse import unquote

# hàm đọc dữ liệu từ đường dẫn được cung cấp và trả về dữ liệu dưới dạng dictionary
def doc_du_lieu_crawl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Tìm và loại bỏ phần mục lục trong nội dung văn bản nếu có
def loai_bo_muc_luc(noi_dung):
    match_muc_luc = re.search(r"M[UỤ]C\s*L[UỤ]C", noi_dung, re.IGNORECASE)
    if not match_muc_luc:
        return noi_dung
    # Tìm "Điều 1" hoặc "Chương I/1" đầu tiên xuất hiện SAU vị trí "MỤC LỤC"
    match_noi_dung_that = re.search(
        r"(Ch[uư][ơo]ng\s+[IVXLC\d]+|Điều\s+1\b)", 
        noi_dung[match_muc_luc.end():]
    )
    if match_noi_dung_that:
        vi_tri_bat_dau = match_muc_luc.end() + match_noi_dung_that.start()
        return noi_dung[:match_muc_luc.start()] + noi_dung[vi_tri_bat_dau:]
    return noi_dung  # không tìm thấy nội dung thật -> giữ nguyên, an toàn hơn là xóa nhầm

def chia_nho_van_ban(text, chunk_size=800, chunk_overlap=100):
    separators = ['\n\n', '\n', '. ', ', ', ' ']
    
    def _split(text, separators):
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
            
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
                
        if separator != "":
            splits = text.split(separator)
            # Re-attach separator to avoid losing characters
            splits = [s + separator for s in splits[:-1]] + [splits[-1]]
        else:
            splits = list(text)
            
        good_splits = []
        for s in splits:
            if len(s) > chunk_size and len(separators) > 1:
                next_separators = separators[1:] if separator in separators else separators
                good_splits.extend(_split(s, next_separators))
            else:
                good_splits.append(s)
        return good_splits

    splits = _split(text, separators)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for s in splits:
        if current_length + len(s) > chunk_size and current_length > 0:
            chunks.append("".join(current_chunk).strip())
            
            # Tạo phần overlap
            overlap_chunk = []
            overlap_length = 0
            for item in reversed(current_chunk):
                if overlap_length + len(item) > chunk_overlap and overlap_length > 0:
                    break
                overlap_chunk.insert(0, item)
                overlap_length += len(item)
            
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(s)
        current_length += len(s)
        
    if current_chunk:
        chunks.append("".join(current_chunk).strip())
        
    return [c for c in chunks if c]

def phan_tach_theo_chuong(noi_dung):
    pattern_chuong = r"(?:\n|^)(Ch[uư][ơo]ng\s+[IVXLC\d]+[\.\s:]*[^\n]*)"
    parts = re.split(pattern_chuong, noi_dung)
    if len(parts) == 1:
        return [("", noi_dung)]  # không có Chương -> coi cả văn bản là 1 khối
    ket_qua = [("", parts[0])] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        tieu_de_chuong = parts[i].strip()
        noi_dung_chuong = parts[i+1] if i+1 < len(parts) else ""
        ket_qua.append((tieu_de_chuong, noi_dung_chuong))
    return ket_qua

def phan_tach_theo_dieu(noi_dung, ten_tai_lieu, url_nguon, chuong=""):

    # chứa các doạn văn bản được tách ra từ nội dung
    chunks = []

    # Tìm kiếm tất cả các điều trong nội dung
    pattern = r"(?:\n|^)(Điều\s+\d+[\.\s:]?)"

    # Tách nội dung thành các phần dựa trên mẫu regex
    parts = re.split(pattern, noi_dung)

    if len(parts) == 1:
        text_chunks = chia_nho_van_ban(noi_dung, chunk_size=800, chunk_overlap=100)
        
        for idx, text_chunk in enumerate(text_chunks, 1):
            chunk = {
                "chunk_id": f"{ten_tai_lieu}_Doan_{idx}",
                "article_id": f"Doan_{idx}",
                "source": ten_tai_lieu,
                "url": url_nguon,
                "text": re.sub(r'\n+', '\n', text_chunk).strip()
            }
            if chuong:
                chunk["chuong"] = chuong
            chunks.append(chunk)
            
        return chunks

    # Nếu phần đầu tiên không rỗng, thêm nó vào danh sách chunks với article_id là "LoiNoiDau"
    phan_mo_dau = parts[0].strip()
    if phan_mo_dau:
        chunk = {
            "chunk_id": f"{ten_tai_lieu}_LoiNoiDau",
            "article_id": "LoiNoiDau",
            "source": ten_tai_lieu,
            "url": url_nguon,
            "text": re.sub(r'\n+', '\n', phan_mo_dau).strip()
        }
        if chuong:
            chunk["chuong"] = chuong
        chunks.append(chunk)

    # Duyệt qua các phần còn lại, mỗi điều sẽ có tiêu đề và nội dung điều
    for i in range(1, len(parts), 2):
        tieu_de = parts[i].strip()

        if i + 1 < len(parts):
            noi_dung_dieu = parts[i + 1].strip()
        else:
            noi_dung_dieu = ""

        # Trích xuất tiêu đề
        so_dieu_match = re.search(r"\d+", tieu_de)
        if so_dieu_match:
            so_dieu = so_dieu_match.group()
        else:
            so_dieu = "X"
        article_id = f"Điều_{so_dieu}"

        # Kết hợp tiêu đề và nội dung điều để 
        noi_dung = f"{tieu_de} {noi_dung_dieu}".strip()
        noi_dung = re.sub(r'\n+', '\n', noi_dung)  # Loại bỏ các dòng trống thừa

        # Biến thành cấu trúc dictionary và thêm vào danh sách chunks
        chunk = {
            "chunk_id": f"{ten_tai_lieu}_{article_id}",
            "article_id": article_id,
            "source": ten_tai_lieu,
            "url": url_nguon,
            "text": noi_dung
        }
        if chuong:
            chunk["chuong"] = chuong
        chunks.append(chunk)

    return chunks

# Hàm tạo và lưu FAISS vector database từ các chunks
def tao_luu_vector_db(chunks, folder_luu):

    print ("\n Tải mô hình BAAI/bge-m3 và lưu vector embeddings vào FAISS database...")

    # tải mô hình SentenceTransformer để chuyển đổi văn bản thành embeddings
    model = SentenceTransformer('BAAI/bge-m3')

    # lấy nôi dung từ các chunks
    texts = [chunk["text"] for chunk in chunks]

    print(f"Đang tạo embeddings cho {len(texts)} chunks")

    # dùng mô hình để tạo embeddings cho các văn bản, đồng thời chuẩn hóa embeddings để có độ dài bằng 1
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=16)

    # xác định số chiều của embeddings
    dim = embeddings.shape[1]

    print("Đang tạo và lưu FAISS Vector Database")

    # tạo FAISS index với khoảng cách cosine
    index = faiss.IndexFlatIP(dim)

    # thêm embeddings vào FAISS index
    index.add(embeddings)

    if not os.path.exists(folder_luu):
        os.makedirs(folder_luu)

    faiss.write_index(index, os.path.join(folder_luu, "faiss_index.bin"))

    # lưu metadata (các chunks) vào file pickle để có thể truy xuất sau này
    with open(os.path.join(folder_luu, "metadata.pkl"), 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Đã lưu FAISS index và metadata vào thư mục: {folder_luu}")


if __name__ == "__main__":
    # Lấy đường dẫn tuyệt đối của thư mục chứa file ingestion.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Đi ngược ra thư mục gốc, sau đó vào thư mục data/raw để tìm file
    file_path = os.path.join(current_dir, "..", "data", "raw", "ket_qua_crawl.json") 
    
    try:
        data = doc_du_lieu_crawl(file_path)
        print(f"Đã đọc thành công {len(data)} bản ghi từ file JSON.")
    except FileNotFoundError:
        print(f"Vẫn không tìm thấy file tại: {file_path}")
        sys.exit(1)

    tat_ca_chunks = []
    
    # Dùng dictionary để gom nhóm file đính kèm theo url_file nhằm deduplicate và lưu references
    files_dinh_kem = {}
    
    # Duyệt qua toàn bộ dữ liệu crawl để tách chunk
    for bai_viet in data:
        url_bai_viet = bai_viet.get("url", "")

        # 1. Tách chunk cho phần nội dung text của trang web
        if bai_viet.get("noi_dung"):
            # Lấy tên file hoặc đường dẫn làm source, nếu không có lấy tiêu đề bài viết
            url_str = bai_viet.get("url", "")
            source = unquote(url_str.rstrip('/').split('/')[-1]) or bai_viet.get("tieu_de", "web_page")

            # Loại bỏ phần mục lục nếu có
            noi_dung_sach = loai_bo_muc_luc(bai_viet["noi_dung"])
            
            danh_sach_chuong = phan_tach_theo_chuong(noi_dung_sach)
            for chuong_title, noi_dung_chuong in danh_sach_chuong:
                chunks_web = phan_tach_theo_dieu(noi_dung_chuong, source, url_bai_viet, chuong=chuong_title)
                # Cập nhật references cho bài viết web
                for chunk in chunks_web:
                    chunk["references"] = [url_bai_viet]
                tat_ca_chunks.extend(chunks_web)
            
        # 2. Gom nhóm thông tin file đính kèm (PDF, DOCX)
        for file_dk in bai_viet.get("van_ban_dinh_kem", []):
            url_file = file_dk.get("url_file", url_bai_viet)  # Nếu không có url_file, dùng url của bài viết
            if url_file not in files_dinh_kem:
                files_dinh_kem[url_file] = {
                    "source_file": file_dk.get("ten_file", "unknown_file"),
                    "noi_dung_file": file_dk.get("noi_dung", ""),
                    "references": [url_bai_viet]
                }
            else:
                if url_bai_viet not in files_dinh_kem[url_file]["references"]:
                    files_dinh_kem[url_file]["references"].append(url_bai_viet)
                    
    # 3. Tiến hành tách chunk cho các file đính kèm (đã deduplicate)
    for url_file, file_info in files_dinh_kem.items():
        noi_dung_file = file_info["noi_dung_file"]
        if noi_dung_file:
            # Loại bỏ phần mục lục nếu có
            noi_dung_file = loai_bo_muc_luc(noi_dung_file)
            danh_sach_chuong = phan_tach_theo_chuong(noi_dung_file)
            for chuong_title, noi_dung_chuong in danh_sach_chuong:
                chunks_file = phan_tach_theo_dieu(noi_dung_chuong, file_info["source_file"], url_file, chuong=chuong_title)
                # Gắn danh sách references vào từng chunk của file
                for chunk in chunks_file:
                    chunk["references"] = file_info["references"]
                tat_ca_chunks.extend(chunks_file)
            
    print(f"Tổng số chunks đã phân tách: {len(tat_ca_chunks)}")

    # # In thử các chunk từ nhánh Fallback (bất kể URL là gì)
    # print("\n--- XEM TRƯỚC CHUNK TỪ NHÁNH FALLBACK (KHÔNG CÓ ĐIỀU) ---")
    # dem_in = 0
    # for c in tat_ca_chunks:
    #     if "Doan_" in c["article_id"]: # Chỉ lấy các chunk từ nhánh fallback
    #         print(json.dumps(c, ensure_ascii=False, indent=4))
    #         dem_in += 1
    #         if dem_in >= 3: # In 3 chunk để đánh giá
    #             break
                
    # if dem_in == 0:
    #     print("Toàn bộ dữ liệu đều có chứa chữ 'Điều X', không có trang nào rơi vào fallback.")
    # print("------------------------------\n")
    
    # Tạo và lưu FAISS vector database
    db_folder = os.path.join(current_dir, "..", "data", "vector_db")
    tao_luu_vector_db(tat_ca_chunks, db_folder)




