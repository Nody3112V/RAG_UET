import json
import re
import os
import pickle
import faiss
import sys
import hashlib
import torch
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
    return noi_dung  # không tìm thấy nội dung thật -> giữ nguyên

# Cắt nhỏ văn bản đệ quy (Recursive Character Text Splitter)
# Đảm bảo chia text thành các đoạn nhỏ không vượt quá chunk_size
# đồng thời giữ lại phần trùng lặp (overlap) giữa các khối.
def chia_nho_van_ban(text, chunk_size=800, chunk_overlap=100):
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    # Chia văn bản tránh việc có 1 đoạn quá dài
    def split_text(text, separators):
        # Tìm ký tự phân tách phù hợp nhất
        separator = separators[-1]
        next_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                next_separators = separators[i + 1:] #kí tự để cắt tiếp theo nếu văn bản còn quá dài
                break

        # Tách văn bản theo kí tự tách tìm được
        splits = list(text) if separator == "" else text.split(separator)
        
        chunks = []
        current_chunk = []
        current_len = 0
        _separator = separator if separator else ""
        
        # lặp qua các đoạn vừa tách
        for s in splits:
            if not s: # Bỏ qua các chuỗi rỗng
                continue
                
            if len(s) > chunk_size: # 1 đoạn quá dài so với chunk_size 
                # Nếu chunk hiện tại đang có dữ liệu, lưu lại thành 1 chunk hoàn chỉnh
                if current_chunk:
                    chunks.append(_separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Gọi đệ quy để cắt đoạn dài này
                if next_separators:
                    chunks.extend(split_text(s, next_separators))
                else:
                    # Cắt theo chunk_size
                    for i in range(0, len(s), chunk_size):
                        chunks.append(s[i:i+chunk_size])
            else:
                s_len = len(s)
                sep_len = len(_separator) if current_chunk else 0
                
                # Nếu thêm phần tử này vào làm vượt quá chunk_size
                if current_len + sep_len + s_len > chunk_size and current_chunk:
                    # Đóng gói chunk
                    chunks.append(_separator.join(current_chunk))
                    
                    # Giữ lại một phần (overlap) cho chunk tiếp theo
                    overlap_chunk = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        p_len = len(p)
                        added_len = p_len + (len(_separator) if overlap_chunk else 0)
                        if overlap_len + added_len > chunk_overlap:
                            break
                        overlap_chunk.insert(0, p)
                        overlap_len += added_len
                        
                    current_chunk = overlap_chunk
                    current_len = overlap_len
                    sep_len = len(_separator) if current_chunk else 0
                    
                current_chunk.append(s)
                current_len += s_len + sep_len
                
        if current_chunk:
            chunks.append(_separator.join(current_chunk))
            
        return chunks

    return split_text(text, separators)

# tạo ID riêng cho từng chunk
def tao_chunk_id(ten_tai_lieu, chuong, article_id):
    parts = [ten_tai_lieu]
    if chuong:
        chuong_short = re.sub(r'[^a-zA-Z0-9IVXLC]', '', chuong.split('.')[0].replace(' ', ''))
        parts.append(chuong_short)
    parts.append(article_id)
    base_id = "_".join(parts)
    # tạo mã băm phân biệt các chunk trong trường hợp 1 điều khoản bị chia thành nhiều chunk
    hash_id = hashlib.md5(base_id.encode('utf-8')).hexdigest()[:8]
    return f"{base_id}_{hash_id}"

def phan_tach_theo_khoan(noi_dung_dieu, tieu_de_dieu, ten_tai_lieu, url_nguon, chuong="", chunk_size=800, chunk_overlap=100):
    pattern_khoan = r'(?:\n|^)(\d+)\.\s+' # (?:\n|^) -> Bắt đầu dòng mới hoặc bắt đầu đoạn -> (\d+) -> Một dãy số -> \. -> Dấu chấm -> \s+ -> Dấu cách.
    khoan_parts = re.split(pattern_khoan, noi_dung_dieu)
    if len(khoan_parts) <= 1:
        return None
    chunks = []
    mo_dau = khoan_parts[0].strip() # phần mở đầu trước điều khoản 1 (nếu có)
    current_text = f"{tieu_de_dieu}\n{mo_dau}" if mo_dau else tieu_de_dieu
    for i in range(1, len(khoan_parts), 2): # nhảy 2 do 1 điều khoản gồm 2 phần: số điều khoản và nội dung điều khoản
        so_khoan = khoan_parts[i]
        noi_dung_khoan = khoan_parts[i + 1] if i + 1 < len(khoan_parts) else ""
        khoan_text = f"{so_khoan}. {noi_dung_khoan.strip()}"

        # Nếu chunk còn trống
        if len(current_text) + len(khoan_text) + 1 <= chunk_size:
            current_text += f"\n{khoan_text}"
        else: # chunk đã đầy
            if current_text.strip():
                chunks.append(current_text.strip())
            current_text = f"{tieu_de_dieu}\n{khoan_text}"
    # thêm chunk cuối cùng
    if current_text.strip():
        chunks.append(current_text.strip())
    return chunks

# Chia chunk đối với các tài liệu có cấu trúc heading
def phan_tach_theo_heading(noi_dung, ten_tai_lieu, url_nguon, chuong="", chunk_size=800, chunk_overlap=100):
    pattern = r'(?:\n|^)(#{1,3}\s+[^\n]+)' # Bắt đầu dòng mới hoặc bắt đầu đoạn -> (#{1,3}) -> Các heading từ H1 đến H3 -> (\s+) -> Dấu cách -> (\n+) -> Xuống dòng
    parts = re.split(pattern, noi_dung)

    # Nếu không có heading thì chia chunk theo bình thường
    if len(parts) <= 1:
        return chia_nho_van_ban(noi_dung, chunk_size, chunk_overlap)

    # khởi tạo các biến
    chunks = []
    current_heading = ""
    current_text = ""

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        
        # Nếu là heading 
        if re.match(r'^#{1,3}\s+', part):
            if current_text.strip(): # nếu chunk đã có dữ liệu
                full_text = f"{current_heading}\n{current_text}" if current_heading else current_text
                if len(full_text) > chunk_size: # nếu chunk quá dài
                    chunks.extend(chia_nho_van_ban(full_text, chunk_size, chunk_overlap))
                else:
                    chunks.append(full_text.strip())
            current_heading = part
            current_text = ""
        else: # không phải heading
            current_text += "\n" + part

    # Thêm chunk cuối cùng
    if current_text.strip():
        full_text = f"{current_heading}\n{current_text}" if current_heading else current_text
        if len(full_text) > chunk_size:
            chunks.extend(chia_nho_van_ban(full_text, chunk_size, chunk_overlap))
        else:
            chunks.append(full_text.strip())
    return chunks

# Vẽ biểu đồ thống kê
def thong_ke_chunks(chunks):
    lengths = [len(c["text"]) for c in chunks]
    if not lengths:
        print("Không có chunk nào để thống kê.")
        return
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ CHUNKS:")
    print(f"   - Tổng số chunks: {len(chunks)}")
    print(f"   - Kích thước trung bình: {sum(lengths) / len(lengths):.0f} ký tự")
    print(f"   - Nhỏ nhất: {min(lengths)} ký tự")
    print(f"   - Lớn nhất: {max(lengths)} ký tự")
    
    ranges = [(0, 100), (100, 300), (300, 500), (500, 800), (800, float('inf'))]
    for low, high in ranges:
        count = sum(1 for l in lengths if low <= l < high)
        label = f"{low}-{high}" if high != float('inf') else f"{low}+"
        print(f"   - [{label}]: {count} chunks ({count/len(chunks)*100:.1f}%)")
    print(f"{'='*60}")

# Hàm tách văn bản theo chương
def phan_tach_theo_chuong(noi_dung):

    # Mẫu regex để tìm các tiêu đề chương trong văn bản
    pattern_chuong = r"(?:\n|^)(Ch[uư][ơo]ng\s+[IVXLC\d]+[\.\s:]*[^\n]*)"
    # (?:\n|^) -> bắt đầu dòng mới hoặc bắt đầu đoạn
    # (Ch[uư][ơo]ng\s+[IVXLC\d]+[\.\s:]*[^\n]*) -> tiêu đề chương (Chương + số chương + nội dung chương)

    # chia nhỏ văn bản dựa trên mẫu regex, giữ lại tiêu đề chương và nội dung của chương
    parts = re.split(pattern_chuong, noi_dung)

    if len(parts) == 1:
        return [("", noi_dung)]  # không có Chương -> coi cả văn bản là 1 khối

    # Lưu phần nội dung trước chương đầu tiên
    ket_qua = [("", parts[0])] if parts[0].strip() else []

    # Duyệt qua các phần còn lại, mỗi chương sẽ có tiêu đề và nội dung chương
    for i in range(1, len(parts), 2):
        tieu_de_chuong = parts[i].strip()
        noi_dung_chuong = parts[i+1] if i+1 < len(parts) else ""
        ket_qua.append((tieu_de_chuong, noi_dung_chuong))
    return ket_qua

# Ham tách văn bản theo điều (kết hợp với cắt nhỏ đệ quy)
def phan_tach_theo_dieu(noi_dung, ten_tai_lieu, url_nguon, chuong="", chunk_size=800, chunk_overlap=100):

    # chứa các doạn văn bản được tách ra từ nội dung
    chunks = []

    # Tìm kiếm tất cả các điều trong nội dung 
    pattern = r"(?:\n|^)(?:#{1,4}\s+)?(?:\*{1,2})?(Điều\s+\d+[\.\s:\-\)]*[^\n]*)(?:\*{1,2})?"
    # đàu dòng -> (\n|^)
    # heading (từ H1 đến H4) -> (?:#{1,4}\s+)?
    # bold -> (?:\*{1,2})?
    # điều ->Điều
    # số điều -> \d+
    # khoảng cách -> [\.\s:\-\)]*
    # nội dung điều -> [^\n]*
    # dấu * ở cuối -> (?:\*{1,2})?

    # Tách nội dung thành các phần dựa trên mẫu regex
    # parts có dạng: [phần mở đầu, điều 1, nội dung điều 1, điều 2, nội dung điều 2, ...]
    parts = re.split(pattern, noi_dung)

    # Nếu không có điều nào được tìm thấy, chia nhỏ theo heading
    if len(parts) == 1:
        # chia nhỏ theo heading
        text_chunks = phan_tach_theo_heading(noi_dung, ten_tai_lieu, url_nguon, chuong, chunk_size, chunk_overlap)
        
        for idx, text_chunk in enumerate(text_chunks, 1):
            # không có điều nên gắn ID theo đoạn
            article_id = f"Đoạn_{idx}"
            chunk_id = tao_chunk_id(ten_tai_lieu, chuong, article_id)
            chunk = {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "source": ten_tai_lieu,
                "url": url_nguon,
                "text": re.sub(r'\n+', '\n', text_chunk).strip() # xoá dòng trống
            }
            if chuong:
                chunk["chuong"] = chuong
            chunks.append(chunk)
            
        return chunks

    # Nếu phần đầu tiên không rỗng, thêm nó vào danh sách chunks với article_id là "LoiNoiDau"
    phan_mo_dau = parts[0].strip()
    if phan_mo_dau:
        text_chunks = chia_nho_van_ban(phan_mo_dau, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, text_chunk in enumerate(text_chunks, 1):
            article_id = "LoiNoiDau" if len(text_chunks) == 1 else f"LoiNoiDau_Phan_{idx}"
            chunk_id = tao_chunk_id(ten_tai_lieu, chuong, article_id)
            chunk = {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "source": ten_tai_lieu,
                "url": url_nguon,
                "text": re.sub(r'\n+', '\n', text_chunk).strip()
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

        # Trích xuất số điều
        so_dieu_match = re.search(r"\d+", tieu_de)
        if so_dieu_match:
            so_dieu = so_dieu_match.group()
        else:
            so_dieu = "X"
        base_article_id = f"Điều_{so_dieu}"

        # Kết hợp tiêu đề và nội dung điều để 
        noi_dung = f"{tieu_de}\n{noi_dung_dieu}".strip()
        
        # Thử phân tách theo khoản
        khoan_chunks = phan_tach_theo_khoan(noi_dung_dieu, tieu_de, ten_tai_lieu, url_nguon, chuong, chunk_size, chunk_overlap)
        
        text_chunks = []
        if khoan_chunks is not None:
            for kc in khoan_chunks:
                if len(kc) > chunk_size: # Nếu khoản quá dài
                    text_chunks.extend(chia_nho_van_ban(kc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                else: # Nếu khoản vừa đủ
                    text_chunks.append(kc)
        else: # không cắt theo khoản được
            text_chunks = chia_nho_van_ban(noi_dung, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        for idx, text_chunk in enumerate(text_chunks, 1):
            article_id = base_article_id if len(text_chunks) == 1 else f"{base_article_id}_Phan_{idx}"
            chunk_id = tao_chunk_id(ten_tai_lieu, chuong, article_id)
            
            # Biến thành cấu trúc dictionary và thêm vào danh sách chunks
            chunk = {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "source": ten_tai_lieu,
                "url": url_nguon,
                "text": re.sub(r'\n+', '\n', text_chunk).strip()
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

    #tạo embeddings cho các văn bản, đồng thời chuẩn hóa embeddings để có độ dài bằng 1
    batch_size = 32 if torch.cuda.is_available() else 16
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=batch_size)

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

    # lưu metadata (các chunks) vào file pickle 
    with open(os.path.join(folder_luu, "metadata.pkl"), 'wb') as f:
        pickle.dump(chunks, f)

    # Lưu chunks ra JSON để kiểm tra
    chunks_json_path = os.path.join(folder_luu, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(chunks)} chunks vào: {chunks_json_path}")

    print(f"Đã lưu FAISS index, metadata và chunks.json vào thư mục: {folder_luu}")


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
                # Cập nhật các đường dẫn nguồn cho bài viết web
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
                    
    # 3. Tiến hành tách chunk cho các file đính kèm
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
            
    print(f"Tổng số chunks đã phân tách ban đầu: {len(tat_ca_chunks)}")

    # Lọc chunks quá nhỏ (Issue 7)
    MIN_CHUNK_LENGTH = 20
    tat_ca_chunks = [c for c in tat_ca_chunks if len(c["text"].strip()) >= MIN_CHUNK_LENGTH]
    print(f"Tổng số chunks sau khi lọc (< {MIN_CHUNK_LENGTH} ký tự): {len(tat_ca_chunks)}")

    # Thống kê phân bổ (Issue 6)
    thong_ke_chunks(tat_ca_chunks)

    # Tạo và lưu FAISS vector database
    db_folder = os.path.join(current_dir, "..", "data", "vector_db")
    tao_luu_vector_db(tat_ca_chunks, db_folder)




