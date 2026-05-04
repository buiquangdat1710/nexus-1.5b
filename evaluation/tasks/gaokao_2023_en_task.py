import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness, normalize_math_string


class GaoKao2023EnTask(BaseEvalTask):
    """
    GaoKao 2023 EN Benchmark (MARIO-Math-Reasoning/Gaokao2023-Math-En).
    385 bài toán dịch sang tiếng Anh từ Kỳ thi tuyển sinh đại học Trung Quốc 2023,
    AMC 2023, và ACT 2023.
    Columns: question, answer, source, lang, sourcename, id.
    Chỉ có split 'train' (dùng làm test set).
    """

    def __init__(self, num_shots: int = 4):
        super().__init__(
            dataset_name="MARIO-Math-Reasoning/Gaokao2023-Math-En",
            split="train",  # Dataset chỉ có split 'train'
            num_shots=num_shots
        )

        self.system_prompt = (
            "You are a mathematics expert. Solve the following problem step by step, "
            "and put your final answer in \\boxed{}."
        )

        self.golden_shots = [
            {
                "question": "If $z = 1 + i$, then $|z^2 - 2z| = $",
                "answer": "We compute $z^2 = (1+i)^2 = 1 + 2i + i^2 = 2i$.\n"
                          "Then $z^2 - 2z = 2i - 2(1+i) = 2i - 2 - 2i = -2$.\n"
                          "So $|z^2 - 2z| = |-2| = \\boxed{2}$.\n"
                          "The answer is: 2."
            },
            {
                "question": "Given the function $f(x) = x^3 - 3x + 1$, find the number of zeros of $f(x)$ on the interval $[-2, 2]$.",
                "answer": "We evaluate: $f(-2) = -8 + 6 + 1 = -1 < 0$, $f(-1) = -1 + 3 + 1 = 3 > 0$, "
                          "$f(1) = 1 - 3 + 1 = -1 < 0$, $f(2) = 8 - 6 + 1 = 3 > 0$.\n"
                          "By the Intermediate Value Theorem, there are zeros in $(-2,-1)$, $(-1,1)$, and $(1,2)$.\n"
                          "Since $f'(x) = 3x^2 - 3 = 3(x-1)(x+1)$, $f$ has exactly one local max and one local min, "
                          "so there are exactly $\\boxed{3}$ zeros.\n"
                          "The answer is: 3."
            },
            {
                "question": "In triangle $ABC$, the sides opposite to angles $A$, $B$, $C$ are $a$, $b$, $c$ respectively. "
                            "If $a = 2$, $b = 3$, and $\\cos C = \\frac{1}{4}$, find the area of triangle $ABC$.",
                "answer": "Using $\\cos C = 1/4$, we get $\\sin C = \\sqrt{1 - 1/16} = \\sqrt{15}/4$.\n"
                          "Area $= \\frac{1}{2}ab\\sin C = \\frac{1}{2} \\cdot 2 \\cdot 3 \\cdot \\frac{\\sqrt{15}}{4} "
                          "= \\frac{3\\sqrt{15}}{4}$.\n"
                          "So the area is $\\boxed{\\frac{3\\sqrt{15}}{4}}$.\n"
                          "The answer is: $\\frac{3\\sqrt{15}}{4}$."
            },
            {
                "question": "If $\\log_2 a + \\log_2 b \\geq 1$, what is the minimum value of $\\frac{1}{a} + \\frac{1}{b}$?",
                "answer": "From $\\log_2 a + \\log_2 b \\geq 1$, we get $ab \\geq 2$.\n"
                          "By AM-HM inequality: $\\frac{1}{a} + \\frac{1}{b} \\geq \\frac{4}{a+b}$.\n"
                          "By AM-GM: $a + b \\geq 2\\sqrt{ab} \\geq 2\\sqrt{2}$.\n"
                          "Also $\\frac{1}{a} + \\frac{1}{b} = \\frac{a+b}{ab} \\geq \\frac{2\\sqrt{ab}}{ab} "
                          "= \\frac{2}{\\sqrt{ab}} \\leq \\frac{2}{\\sqrt{2}} = \\sqrt{2}$.\n"
                          "The minimum value is $\\boxed{\\sqrt{2}}$.\n"
                          "The answer is: $\\sqrt{2}$."
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        try:
            ds = load_dataset(self.dataset_name, split=self.split)
            examples = [{"question": item["question"]} for item in ds]
            gold_answers = [item["answer"] for item in ds]
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
        #  \boxed{}
        is_correct = check_correctness(prediction, gold_answer)
        if is_correct:
            return True

        # Fallback: "The answer is: ..."
        fallback_pattern = r"The answer is:?\s*\$?([^\$\n]+)\$?"
        matches = re.findall(fallback_pattern, prediction, re.IGNORECASE)
        if matches:
            pred_ans = matches[-1].strip().rstrip(".")
            return normalize_math_string(pred_ans) == normalize_math_string(gold_answer)

        # Fallback: "Final answer: ..."
        final_pattern = r"[Ff]inal [Aa]nswer:?\s*(?:[Tt]he final answer is\s*)?\$?([^\$\n]+)\$?"
        final_matches = re.findall(final_pattern, prediction)
        if final_matches:
            pred_ans = final_matches[-1].strip().rstrip(".")
            return normalize_math_string(pred_ans) == normalize_math_string(gold_answer)

        # Fallback số: so sánh với tolerance
        try:
            gold_num = float(gold_answer.replace(",", ""))
            nums = re.findall(r'-?[\d]+\.?[\d]*', prediction.replace(",", ""))
            if nums:
                pred_num = float(nums[-1])
                if gold_num != 0:
                    return abs(pred_num - gold_num) / abs(gold_num) < 0.02
                else:
                    return abs(pred_num - gold_num) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass

        return False
