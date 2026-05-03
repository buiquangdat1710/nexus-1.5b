import re
from typing import Optional

def extract_boxed_answer(text: str) -> Optional[str]:
    """Trích xuất kết quả nằm trong \boxed{}."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None

def normalize_math_string(s: str) -> str:
    """Chuẩn hóa chuỗi latex để so sánh chính xác."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    # Loại bỏ các ký tự không ảnh hưởng đến giá trị toán học
    for ch in ("$", "\\,", "\\!", "{", "}"):
        s = s.replace(ch, "")
    return s

def check_correctness(prediction: str, gold_answer: str) -> bool:
    """So sánh kết quả của model với đáp án gốc."""
    pred_boxed = extract_boxed_answer(prediction)
    if not pred_boxed:
        return False
        
    return normalize_math_string(pred_boxed) == normalize_math_string(gold_answer)