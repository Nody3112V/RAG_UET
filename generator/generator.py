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
    def generate_answer(self, query, retrieved_chunks, chat_history=None):

        # Nếu mảng retrieved_chunks rỗng, tạo context rỗng
        if not retrieved_chunks:
            context = ""
        else:
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
            "Bạn là trợ lý AI tiếng Việt của UET. Nhiệm vụ của bạn là trả lời câu hỏi CHỈ DỰA VÀO NGỮ CẢNH (Context) được cung cấp.\n"
            "Quy tắc BẮT BUỘC:\n"
            "1. Tuyệt đối chỉ sử dụng tiếng Việt chuẩn, không sử dụng ký tự tiếng Trung hoặc ngôn ngữ khác.\n"
            "2. Đọc thật kỹ NGỮ CẢNH. Nếu câu hỏi liên quan đến ĐIỂM SỐ hoặc ĐIỀU KIỆN, hãy so sánh các con số thật cẩn thận.\n"
            "3. Nếu thông tin trong NGỮ CẢNH không nói rõ hoặc không chứa câu trả lời cho câu hỏi, bạn TUYỆT ĐỐI KHÔNG được tự suy diễn hay phỏng đoán. Bắt buộc phải trả lời đúng một câu: 'Tôi không có đủ dữ liệu để trả lời câu hỏi này'.\n"
            "4. Không đưa ra lời khuyên ngoài lề. Không kết hợp thông tin lộn xộn.\n"
            "5. Luôn giữ nguyên ngữ cảnh của chủ thể được hỏi khi trả lời.\n"
            "6. Hãy tổng hợp thông tin và trả lời người dùng một cách tự nhiên. Tuyệt đối KHÔNG sử dụng các cụm từ như 'Theo tài liệu 1', 'Theo ngữ cảnh cung cấp', hay 'Dựa vào văn bản',...\n"
            "7. Hãy đọc thật kỹ tên của các đối tượng (như tên học bổng, danh hiệu, điều luật) trong câu hỏi và đối chiếu chính xác tuyệt đối với tên trong ngữ cảnh. Không lấy điều kiện của đối tượng này gán cho đối tượng khác.\n"
            "8. Khi phân tích dữ liệu dạng bảng (Markdown table), nếu ô tương ứng với đối tượng được hỏi bị để trống hoặc ghi không có thông tin, bạn phải kết luận là không có thông tin/lịch trình, tuyệt đối KHÔNG ĐƯỢC lấy thông tin của cột/hàng khác gán vào."
        )

        # Tạo danh sách các tin nhắn để cung cấp cho mô hình sinh văn bản
        messages = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử hội thoại vào để xử lý Multi-turn Conversation
        if chat_history:
            # Lấy tối đa 4 tin nhắn gần nhất để làm ngữ cảnh
            for msg in chat_history[-4:]:
                # Lọc bỏ phần nguồn tham khảo nếu có để tối ưu token
                content = msg["content"].split("\n\n**Nguồn tham khảo:**")[0]
                messages.append({"role": msg["role"], "content": content})

        user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:"
        messages.append({"role": "user", "content": user_prompt})

        # Áp dụng mẫu chat để tạo đầu vào cho mô hình sinh văn bản
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
        )

        # Mã hóa đầu vào thành tensor và chuyển sang thiết bị của mô hình
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Sinh văn bản dựa trên đầu vào đã mã hóa
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512, # Tăng số lượng token sinh ra để trả lời đầy đủ thông tin hơn   
            temperature=0.3,# Giảm độ ngẫu nhiên trong quá trình sinh văn bản để tăng tính chính xác
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

        return response

