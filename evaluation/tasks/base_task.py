import abc
from typing import List, Dict, Tuple

class BaseEvalTask(abc.ABC):
    """Lớp nền tảng cho mọi tác vụ đánh giá (Benchmark Task)."""
    
    def __init__(self, dataset_name: str, split: str = "test", num_shots: int = 0):
        self.dataset_name = dataset_name
        self.split = split
        self.num_shots = num_shots
        # Danh sách chứa các ví dụ mẫu mực. Các class con sẽ điền giá trị vào đây.
        self.golden_shots = [] 

    @abc.abstractmethod
    def load_data(self) -> Tuple[List[Dict], List[str]]:
        """
        Tải dữ liệu.
        Returns: Tuple(List[examples], List[gold_answers])
        """
        pass

    @abc.abstractmethod
    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        """Tạo prompt tùy chỉnh theo từng Task."""
        pass

    @abc.abstractmethod
    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:
        """Chấm điểm đúng/sai theo chuẩn của bộ dữ liệu."""
        pass
        
    def generate_few_shots(self) -> List[Dict]:
        """
        Lấy các ví dụ Golden Shots đã được hard-code sẵn trong class con.
        Đảm bảo không lấy vượt quá self.num_shots.
        """
        if not self.golden_shots or self.num_shots == 0:
            return []
            
        return self.golden_shots[:self.num_shots]