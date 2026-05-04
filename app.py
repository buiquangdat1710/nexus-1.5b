# app.py
import streamlit as st
import requests
import json
from datetime import datetime
import re
import html

# Page configuration
st.set_page_config(
    page_title="Math Solver AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS với hỗ trợ LaTeX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-error {
        color: #F44336;
        font-weight: bold;
    }
    .stForm {
        border: 0 !important;
    }
    /* Cải thiện hiển thị LaTeX */
    .katex { 
        font-size: 1.1em !important;
    }
    .step-container {
        margin: 15px 0;
        padding: 10px;
        border-left: 3px solid #4CAF50;
        background-color: #f8f9fa;
    }
    /* Hiển thị boxed answer đẹp hơn */
    .latex-boxed {
        display: block;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-radius: 10px;
        border: 3px solid #FFC107;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Hàm chuyển đổi LaTeX và xử lý HTML
def format_latex_response(text):
    """
    Định dạng response từ model để hiển thị LaTeX đúng cách trong Streamlit
    """
    # 1. Xử lý HTML entities
    text = html.unescape(text)
    
    # 2. Loại bỏ tất cả các thẻ HTML
    text = re.sub(r'<[^>]*>', '', text)
    
    # 3. Sửa các lỗi LaTeX phổ biến
    # Sửa lỗi: \frac -> frac
    text = re.sub(r'\{frac', r'\\frac', text)
    # Sửa lỗi: boxed{ -> \boxed{
    text = re.sub(r'(?<!\\)boxed\{', r'\\boxed{', text)
    # Sửa lỗi: $$$boxed -> \boxed
    text = re.sub(r'\$\$\$boxed\{', r'\\boxed{', text)
    # Sửa lỗi: $boxed -> \boxed
    text = re.sub(r'\$boxed\{', r'\\boxed{', text)
    # Sửa lỗi: \boxed{...}$[3] -> \boxed{...}
    text = re.sub(r'(\\)?boxed\{[^}]*\}\$?\[.*?\]', lambda m: f'\\boxed{{{extract_boxed_content(m.group(0))}}}', text)
    
    # 4. Xử lý LaTeX display: \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 5. Xử lý LaTeX inline: \( ... \) -> $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # 6. Xử lý \boxed{...} đặc biệt
    def process_boxed(match):
        content = match.group(1).strip()
        # Loại bỏ các ký tự lạ
        content = re.sub(r'^\$+|\$+$', '', content)  # Loại bỏ $ ở đầu/cuối
        content = re.sub(r'\[.*?\]', '', content)  # Loại bỏ [3] hay bất kỳ [number] nào
        content = re.sub(r'\\\\', '', content)  # Loại bỏ \\ thừa
        
        # Nếu content trống, trả về chuỗi rỗng
        if not content or content.isspace():
            return ''
        
        # Đảm bảo content là LaTeX hợp lệ
        return f'$$\n\\boxed{{{content}}}\n$$'
    
    # Tìm và xử lý tất cả các boxed
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    text = re.sub(boxed_pattern, process_boxed, text, flags=re.DOTALL)
    
    # 7. Tìm và sửa các lỗi boxed không hoàn chỉnh
    # Pattern cho boxed bị thiếu dấu }
    incomplete_boxed = r'\\boxed\{([^}]*)(?=\n|$)'
    def fix_incomplete_boxed(match):
        content = match.group(1)
        return f'\\boxed{{{content}}}'
    
    text = re.sub(incomplete_boxed, fix_incomplete_boxed, text)
    
    # 8. Sửa các lỗi LaTeX khác
    # Sửa lỗi frac không đúng: \{frac -> \frac
    text = re.sub(r'\\\{frac([^{])', r'\\frac{\1', text)
    # Sửa lỗi dấu ngoặc không khớp
    text = re.sub(r'\\\{', '{', text)
    text = re.sub(r'\\\}', '}', text)
    
    # 9. Đảm bảo các công thức toán học được hiển thị đẹp
    lines = text.split('\n')
    formatted_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Xử lý các bước đánh số (1., 2., ...)
        if re.match(r'^\d+\.', stripped):
            formatted_lines.append(f'\n**{stripped}**')
        # Xử lý dòng có chứa boxed
        elif '\\boxed{' in stripped:
            formatted_lines.append(f'\n{stripped}')
        # Xử lý dòng có công thức display
        elif stripped.startswith('$$') or stripped.endswith('$$'):
            formatted_lines.append(f'\n{stripped}')
        # Xử lý dòng có kết thúc bằng dấu $ (inline LaTeX)
        elif stripped.endswith('$') and not stripped.startswith('$'):
            # Đây có thể là inline LaTeX, thêm newline trước
            formatted_lines.append(f'\n{stripped}')
        else:
            formatted_lines.append(stripped)
    
    text = '\n'.join(formatted_lines)
    
    # 10. Thêm khoảng cách giữa các bước giải
    text = re.sub(r'(\n\*\*\d+\.\*\*)', r'\n\n\1', text)
    
    # 11. Đảm bảo các công thức LaTeX không bị phá vỡ
    text = re.sub(r'(?<!\n)\n\$\$', r'\n\n$$', text)
    text = re.sub(r'\$\$\n(?!\n)', r'$$\n\n', text)
    
    # 12. Xử lý dòng trống thừa
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_boxed_content(boxed_string):
    """Trích xuất nội dung từ chuỗi boxed bị lỗi"""
    # Tìm nội dung giữa { và }
    match = re.search(r'\{([^{}]*)\}', boxed_string)
    if match:
        content = match.group(1)
        # Loại bỏ các ký tự không mong muốn
        content = re.sub(r'\$+', '', content)
        content = re.sub(r'\[.*?\]', '', content)
        return content.strip()
    return ""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = ""
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình")
    
    # API Configuration
    st.subheader("API Configuration")
    api_url = st.text_input(
        "Colab API URL",
        value=st.session_state.api_url,
        placeholder="https://xxxx-xxxx-xxxx.ngrok-free.app",
        help="Nhập URL từ Google Colab (ngrok)"
    )
    
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url.rstrip('/')
        st.success(f"Connected to: {api_url[:50]}..." if len(api_url) > 50 else f"Connected to: {api_url}")
    
    st.divider()
    
    # Model Parameters
    st.subheader("Model Parameters")
    
    reasoning_method = st.selectbox(
        "Reasoning Method",
        ["CoT", "TIR"],
        index=0,
        help="CoT: Chain-of-Thought\nTIR: Tool-Integrated Reasoning"
    )
    
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=128,
        max_value=2048,
        value=1024,
        step=128
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    top_p = st.slider(
        "Top-p (Nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    
    custom_system = st.text_area(
        "Custom System Message (Optional)",
        height=100,
        help="Ghi đè system message mặc định"
    )
    
    st.divider()
    
    # Examples với LaTeX
    st.subheader("📚 Ví dụ")
    examples = [
        "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.",
        "Solve the quadratic equation: $x^2 - 5x + 6 = 0$",
        "What is the derivative of $f(x) = 3x^4 + 2x^2 - 5x + 7$?",
        "Calculate the integral: $\\int (3x^2 + 2x - 1) dx$",
        "Find the limit: $\\lim_{x \\to 0} \\frac{\\sin(x)}{x}$"
    ]
    
    for example in examples:
        if st.button(f"📝 {example[:50]}..." if len(example) > 50 else f"📝 {example}"):
            st.session_state.current_prompt = example
            st.rerun()
    
    st.divider()
    
    # Hiển thị thông tin LaTeX
    with st.expander("ℹ️ Hướng dẫn LaTeX"):
        st.markdown("""
        **Hỗ trợ LaTeX:**
        - `$...$`: Công thức inline (ví dụ: `$x^2$`)
        - `$$...$$`: Công thức display (ví dụ: `$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$`)
        - `\\boxed{...}`: Hiển thị đáp án trong hộp
        
        **Ví dụ:**
        - Phương trình: `$x^2 + y^2 = r^2$`
        - Tích phân: `$$\\int_a^b f(x) dx$$`
        - Đáp án: `\\boxed{x = -1}`
        """)
    
    # Clear chat button
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">🧮 Math Solver AI Assistant</h1>', unsafe_allow_html=True)

# Connection status
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.api_url:
        try:
            # Kiểm tra health endpoint
            response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Connected to {data.get('model', 'API server')}")
            else:
                st.error("❌ Connection failed")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to server")
        except Exception as e:
            st.warning(f"⚠️ Connection error: {str(e)}")
    else:
        st.info("ℹ️ Please enter API URL from Colab in sidebar")

# Chat container
st.markdown("### 💬 Chat")
chat_container = st.container()

# Display chat messages với hỗ trợ LaTeX
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                content = message["content"]
                
                # Format LaTeX
                formatted_content = format_latex_response(content)
                
                # Hiển thị với markdown
                st.markdown(formatted_content)
                
            if "response_data" in message:
                with st.expander("📊 Response Details"):
                    st.json(message["response_data"])

# Input area
with st.form(key="input_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        prompt = st.text_area(
            "Nhập bài toán:",
            value=st.session_state.current_prompt,
            placeholder="Nhập bài toán toán học của bạn ở đây (có thể dùng LaTeX như $x^2 + y^2 = 1$)...",
            height=100,
            key="prompt_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🚀 Gửi", use_container_width=True, type="primary")

# Process input
if submit_button and prompt.strip():
    # Reset current prompt
    st.session_state.current_prompt = ""
    
    # Add user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # Check connection
    if not st.session_state.api_url:
        st.error("Vui lòng nhập API URL trong sidebar trước!")
        st.rerun()
    
    # Show assistant placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Đang xử lý...")
        
        # Prepare request
        request_data = {
            "prompt": prompt,
            "reasoning_method": reasoning_method,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if custom_system.strip():
            request_data["system_message"] = custom_system
        
        try:
            # Send request to Colab API
            response = requests.post(
                f"{st.session_state.api_url}/generate",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Format response với LaTeX
                formatted_response = format_latex_response(raw_response)
                
                # Hiển thị response
                message_placeholder.markdown(formatted_response)
                
                # Add assistant message with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": raw_response,
                    "formatted_content": formatted_response,
                    "response_data": {
                        "model": result.get("model"),
                        "status": result.get("status"),
                        "timestamp": datetime.now().isoformat(),
                        "parameters": result.get("parameters", {})
                    }
                }
                st.session_state.messages.append(assistant_message)
                
            else:
                error_msg = f"❌ Lỗi API: {response.status_code} - {response.text}"
                message_placeholder.markdown(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "⏰ Request timeout. Model might be taking too long."
            message_placeholder.markdown(error_msg)
            
        except Exception as e:
            error_msg = f"❌ Lỗi kết nối: {str(e)}"
            message_placeholder.markdown(error_msg)
    
    # Force rerun để cập nhật giao diện
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Qwen2.5-Math-1.5B-Instruct • Running on Google Colab GPU • Streamlit Frontend</p>
    <p style='font-size: 0.9em;'>📚 Hỗ trợ LaTeX đầy đủ: $...$ cho inline, $$...$$ cho display, \\boxed{} cho đáp án</p>
</div>
""", unsafe_allow_html=True)