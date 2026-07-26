import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"


import streamlit as st
import os
from retriever.retriever import HybridRetriever
from generator.generator import Generator

# cấu hình trang giao diện
st.set_page_config(
    page_title="UET AI Chatbot", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện đẹp hơn
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/vi/thumb/9/93/Logo_UET.svg/1200px-Logo_UET.svg.png", width=150)
    st.title("🎓 UET AI Assistant")
    st.markdown("""
    **Chào mừng bạn đến với Chatbot Tư vấn Sinh viên UET!**
    
    Hệ thống sử dụng công nghệ RAG (Retrieval-Augmented Generation) để tra cứu và trả lời các câu hỏi dựa trên:
    - Sổ tay sinh viên UET
    - Quy chế đào tạo
    - Các văn bản hướng dẫn
    
    *Lưu ý: AI có thể đưa ra câu trả lời chưa chính xác hoàn toàn. Vui lòng đối chiếu với nguồn tham khảo đính kèm.*
    """)
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.caption("© 2026 UET AI Research")

# ----------------- MAIN UI -----------------
st.title("💬 Chatbot tư vấn sổ tay sinh viên UET")
st.markdown("Hãy đặt bất kỳ câu hỏi nào về quy chế, học bổng, điểm rèn luyện... tại UET!")

# chỉ thực hiện 1 lần, những lần sau sẽ lấy kết quả từ cache
@st.cache_resource
def load_ai_models():
    # Đường dẫn đến thư mục vector_db chứa dữ liệu đã được xử lý từ bước Ingestion
    vector_db_path = os.path.join("data", "vector_db")
    # Khởi tạo các mô hình AI: HybridRetriever và Generator
    retriever = HybridRetriever(vector_db_path)
    generator = Generator() 
    return retriever, generator

# Hiển thị thông báo đang tải mô hình AI
with st.spinner("⏳ Đang khởi tạo mô hình AI..."):
    retriever, generator = load_ai_models()

# Kiem tra và khởi tạo lịch sử trò chuyện nếu chưa tồn tại
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Chào bạn! Mình là Chatbot AI của UET. Mình có thể giúp bạn giải đáp các thắc mắc về sổ tay sinh viên, quy chế đào tạo, điểm rèn luyện... Bạn cần mình giúp gì nào?", "sources": []}
    ]

# Hiển thị lịch sử trò chuyện
for message in st.session_state.chat_history:
    avatar_icon = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Xem nguồn tham khảo", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

# Nhận câu hỏi từ người dùng và xử lý
if promt := st.chat_input("Nhập câu hỏi của bạn (VD: Điều kiện nhận học bổng là gì?)..."):

    # Hiển hiện câu hỏi người dùng
    st.session_state.chat_history.append({"role": "user", "content": promt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(promt)

    # Hiển thị thông báo đang tìm kiếm và sinh câu trả lời
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("⏳ Đang tìm kiếm thông tin và suy nghĩ..."):
            
            # Tìm kiếm các chunk liên quan đến câu hỏi của người dùng bằng phương pháp hybrid search
            retrieved_chunks = retriever.hybrid_search(promt, top_k=3)

            # Sinh câu trả lời dựa trên các chunk đã tìm được và câu hỏi của người dùng
            # Lấy lịch sử hội thoại (loại bỏ câu hỏi hiện tại ở cuối mảng)
            history = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else []
            answer = generator.generate_answer(promt, retrieved_chunks, chat_history=history)

            # Trích xuất thông tin nguồn
            sources = []
            for items in retrieved_chunks:
                source_name = items['chunk']['source']
                article = items['chunk']['article_id']
                chuong = items['chunk'].get('chuong', '')
                
                if chuong:
                    citation = f"**{source_name}** ({chuong} - Mục: {article})"
                else:
                    citation = f"**{source_name}** (Mục: {article})"
                    
                if citation not in sources:
                    sources.append(citation)

            # Nếu câu trả lời không đủ dữ liệu, xóa nguồn
            if "Tôi không có đủ dữ liệu" in answer:
                sources = []

            # Hiển thị câu trả lời
            st.markdown(answer)
            if sources:
                with st.expander("📚 Xem nguồn tham khảo", expanded=False):
                    for src in sources:
                        st.markdown(f"- {src}")

    # Cập nhật lịch sử trò chuyện 
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})