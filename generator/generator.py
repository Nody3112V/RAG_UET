import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
# Bỏ comment khi dùng Kaggle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Generator:

    # Khởi tạo Generator 
    def __init__(self, model_name = "Qwen/Qwen2.5-3B-Instruct"):
        print("Đang tải mô hình sinh văn bản")

        # Tải tokenizer từ mô hình Qwen2.5-3B-Instruct
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tải mô hình ngôn ngữ
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" # Tự động phân bổ mô hình trên các thiết bị có sẵn (CPU/GPU)
        )
        print("Đã tải xong mô hình sinh văn bản")

    # Hàm sinh văn bản trả lời dựa trên truy vấn và các chunk được truy xuất
    def generate_answer(self, query, retrieved_chunks, chat_history=None, max_context_tokens=4096):

        # Nếu mảng retrieved_chunks rỗng, trả về câu từ chối ngay
        if not retrieved_chunks:
            return "Tôi không có đủ dữ liệu để trả lời câu hỏi này."
            
        # Ghep các chunk được truy xuất thành một ngữ cảnh duy nhất để cung cấp cho mô hình sinh văn bản
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk['chunk']['source']
            chuong = chunk['chunk'].get('chuong', '')
            nguon_chi_tiet = f"Nguồn: {source}"
            if chuong:
                nguon_chi_tiet += f", {chuong}"
            context_parts.append(f"Tài liệu {i+1} ({nguon_chi_tiet}):\n{chunk['chunk']['text']}")
        
        context = "\n\n".join(context_parts)

        system_prompt = (
            "Bạn là trợ lý AI tiếng Việt của Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội (gọi tắt là UET). Trả lời câu hỏi CHỈ DỰA VÀO NGỮ CẢNH được cung cấp.\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng tiếng Việt. Không dùng tiếng Trung hay ngôn ngữ khác.\n"
            "2. Nếu ngữ cảnh KHÔNG chứa câu trả lời, BẮT BUỘC trả lời: 'Tôi không có đủ dữ liệu để trả lời câu hỏi này'. KHÔNG được tự suy diễn.\n"
            "3. So sánh điểm số, điều kiện, tên đối tượng thật cẩn thận. Không lẫn lộn thông tin giữa các đối tượng khác nhau.\n"
            "4. Trả lời thẳng vào trọng tâm. TUYỆT ĐỐI KHÔNG sử dụng các cụm từ như 'Theo tài liệu', 'Theo ngữ cảnh', 'Tài liệu số', 'Trong đoạn trích'.\n"
            "5. Với bảng Markdown: nếu ô trống thì kết luận không có thông tin, không lấy từ ô khác.\n"
        )

        # Tạo danh sách các tin nhắn để cung cấp cho mô hình sinh văn bản
        messages = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử hội thoại vào để xử lý Multi-turn Conversation
        if chat_history:
            # Lấy tối đa 4 tin nhắn gần nhất để làm ngữ cảnh
            for msg in chat_history[-4:]:
                content = msg["content"]
                # Giới hạn để tiết kiệm token (chỉ lấy 300 ký tự đầu)
                if len(content) > 300:
                    content = content[:300] + "..."
                messages.append({"role": msg["role"], "content": content})

        user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:"
        messages.append({"role": "user", "content": user_prompt})

        # Kiểm tra tổng token trước khi generate
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(self.tokenizer.encode(text))
        
        # Nếu vượt quá giới hạn, cắt bớt context
        max_input = max_context_tokens - 512  # Dành 512 cho output
        while input_tokens > max_input and len(context_parts) > 1:
            context_parts.pop()  # Loại bỏ chunk cuối cùng (ít liên quan nhất)
            context = "\n\n".join(context_parts)
            user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:"
            messages[-1] = {"role": "user", "content": user_prompt}
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_tokens = len(self.tokenizer.encode(text))
        
        # Nếu vẫn vượt quá, cắt bớt chat history
        while input_tokens > max_input and len(messages) > 2:
            messages.pop(1)  # Loại bỏ tin nhắn cũ nhất trong history
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_tokens = len(self.tokenizer.encode(text))

        try:
            # Mã hóa đầu vào thành tensor và chuyển sang thiết bị của mô hình
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    
            # Sinh văn bản dựa trên đầu vào đã mã hóa
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512, # Tăng số lượng token sinh ra để trả lời đầy đủ thông tin hơn   
                    temperature=0.2,# Giảm độ ngẫu nhiên trong quá trình sinh văn bản để tăng tính chính xác
                    top_p=0.9, # Sử dụng top-p sampling để kiểm soát sự đa dạng của văn bản sinh ra
                    repetition_penalty=1.15, # Áp dụng penalty để giảm khả năng lặp lại các token   
                    do_sample=True # Sử dụng sampling để sinh văn bản thay vì greedy decoding    
                )
    
            # Tách các token sinh ra từ đầu vào để lấy phần văn bản mới được sinh ra
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
            ]
    
            # Chuyển đổi các token sinh ra thành văn bản và loại bỏ các token đặc biệt
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
            # Giải phóng bộ nhớ GPU
            del inputs, outputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            return response
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại với câu hỏi ngắn hơn."
        except Exception as e:
            print(f"Lỗi khi sinh câu trả lời: {e}")
            return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại."

