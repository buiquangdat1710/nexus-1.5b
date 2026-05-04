import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness, normalize_math_string


class CollegeMathTask(BaseEvalTask):
    """
    College Math Benchmark (TIGER-Lab/TheoremQA).
    TheoremQA chứa 800 bài toán cấp đại học yêu cầu áp dụng định lý.
    Columns: Question, Answer, Answer_type, theorem_used, subfield.
    Dùng làm proxy cho College Math benchmark.
    """

    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="TIGER-Lab/TheoremQA", split="test", num_shots=num_shots)
        self.system_prompt = (
            "You are a college-level mathematics expert. "
            "Solve the following problem step by step using relevant theorems and formulas. "
            "Put your final answer in \\boxed{}."
        )
        self.golden_shots = [
            {
                "question": "Find the volume of the solid obtained by rotating the region bounded by $y = x^2$, $y = 0$, and $x = 2$ about the y-axis.",
                "answer": "Using the shell method: $V = 2\\pi \\int_0^2 x \\cdot x^2 \\, dx = 2\\pi \\int_0^2 x^3 \\, dx$\n$= 2\\pi [x^4/4]_0^2 = 2\\pi \\cdot 4 = \\boxed{8\\pi}$.\nThe answer is: $8\\pi$."
            },
            {
                "question": "Determine whether the series $\\sum_{n=1}^{\\infty} \\frac{n!}{n^n}$ converges or diverges.",
                "answer": "By the ratio test: $\\frac{a_{n+1}}{a_n} = \\frac{(n+1)! \\cdot n^n}{(n+1)^{n+1} \\cdot n!} = \\frac{n^n}{(n+1)^n} = \\left(\\frac{n}{n+1}\\right)^n \\to e^{-1} < 1$.\nSince the limit is $1/e < 1$, the series $\\boxed{\\text{converges}}$.\nThe answer is: converges."
            },
            {
                "question": "Find the eigenvalues of the matrix $A = \\begin{pmatrix} 4 & 1 \\\\ 2 & 3 \\end{pmatrix}$.",
                "answer": "The characteristic polynomial: $\\det(A - \\lambda I) = (4-\\lambda)(3-\\lambda) - 2 = \\lambda^2 - 7\\lambda + 10 = (\\lambda-5)(\\lambda-2)$.\nThe eigenvalues are $\\boxed{2, 5}$.\nThe answer is: 2, 5."
            },
            {
                "question": "Evaluate the double integral $\\iint_R xy \\, dA$ where $R$ is the region bounded by $y = x$ and $y = x^2$ for $0 \\leq x \\leq 1$.",
                "answer": "$\\int_0^1 \\int_{x^2}^{x} xy \\, dy \\, dx = \\int_0^1 x [y^2/2]_{x^2}^{x} dx = \\int_0^1 x \\cdot \\frac{x^2 - x^4}{2} dx$\n$= \\frac{1}{2} \\int_0^1 (x^3 - x^5) dx = \\frac{1}{2}[x^4/4 - x^6/6]_0^1 = \\frac{1}{2}(1/4 - 1/6) = \\frac{1}{2} \\cdot \\frac{1}{12} = \\boxed{\\frac{1}{24}}$.\nThe answer is: $1/24$."
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        try:
            ds = load_dataset(self.dataset_name, split=self.split)
            examples = [{"question": item["Question"]} for item in ds]
            gold_answers = [str(item["Answer"]) for item in ds]
            return examples, gold_answers
        except Exception as e:
            print(f"Lỗi khi load {self.dataset_name}: {e}. Vui lòng kiểm tra lại dataset.")
            return [], []

    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for ex in few_shot_examples:
            prompt += f"<|im_start|>user\nProblem:\n{ex['question']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\nSolution:\n{ex['answer']}<|im_end|>\n"
        prompt += f"<|im_start|>user\nProblem:\n{problem}\n<|im_end|>\n"
        prompt += f"<|im_start|>assistant\nSolution:\n"
        return prompt

    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:
        if check_correctness(prediction, gold_answer):
            return True
        # Fallback patterns
        for pattern in [r"The answer is:?\s*\$?([^\$\n]+)\$?", r"[Ff]inal [Aa]nswer:?\s*\$?([^\$\n]+)\$?"]:
            matches = re.findall(pattern, prediction, re.IGNORECASE)
            if matches:
                pred_ans = matches[-1].strip().rstrip(".")
                if normalize_math_string(pred_ans) == normalize_math_string(gold_answer):
                    return True
        # Boolean / text answers
        gold_lower = gold_answer.strip().lower()
        if gold_lower in ("true", "false", "yes", "no", "converges", "diverges"):
            if gold_lower in prediction.lower():
                return True
        # Numeric tolerance
        try:
            gold_num = float(gold_answer.replace(",", ""))
            nums = re.findall(r'-?[\d]+\.?[\d]*(?:e[+-]?\d+)?', prediction.replace(",", ""), re.IGNORECASE)
            if nums:
                pred_num = float(nums[-1])
                if gold_num != 0:
                    return abs(pred_num - gold_num) / abs(gold_num) < 0.02
                return abs(pred_num - gold_num) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass
        return False
