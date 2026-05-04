import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness, normalize_math_string


class MinervaMathTask(BaseEvalTask):
    """
    Minerva Math Benchmark (math-ai/minervamath).
    272 bài toán STEM cấp đại học (vật lý, toán, kỹ thuật) từ MIT OpenCourseWare.
    Đáp án thường là số hoặc biểu thức toán học ngắn.
    """

    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="math-ai/minervamath", split="test", num_shots=num_shots)

        self.system_prompt = (
            "You are a STEM expert. Solve the following problem step by step. "
            "Put your final numerical or symbolic answer in \\boxed{}."
        )

        self.golden_shots = [
            {
                "question": "A star has a measured parallax of $0.01^{\\prime \\prime}$, that is, $0.01$ arcseconds. How far away is it, in parsecs?",
                "answer": "By definition, parallax $p$ (in arcseconds) and distance $d$ (in parsecs) are related by $d = 1/p$.\n"
                          "Therefore $d = 1 / 0.01 = \\boxed{100}$ parsecs.\n"
                          "The answer is: 100."
            },
            {
                "question": "A particular star has an absolute magnitude $M=-7$. If this star is observed in a galaxy that is at a distance of $3 \\mathrm{Mpc}$, what will its apparent magnitude be?",
                "answer": "Using the distance modulus formula: $m = M + 5\\log_{10}(d/10\\,\\text{pc})$\n"
                          "$m = -7 + 5\\log_{10}(3 \\times 10^6 / 10) = -7 + 5\\log_{10}(3 \\times 10^5)$\n"
                          "$= -7 + 5 \\times 5.477 = -7 + 27.39 = \\boxed{20.39}$.\n"
                          "The answer is: 20.39."
            },
            {
                "question": "If the Sun's absolute magnitude is $+5$, find the luminosity of a star of magnitude $0$ in ergs/s. A useful constant: the luminosity of the sun is $3.83 \\times 10^{33}$ ergs/s.",
                "answer": "The difference in magnitude is $\\Delta m = 5 - 0 = 5$.\n"
                          "A difference of 5 magnitudes corresponds to a factor of 100 in brightness.\n"
                          "Therefore $L = 100 \\times 3.83 \\times 10^{33} = \\boxed{3.83e35}$ ergs/s.\n"
                          "The answer is: 3.83e35."
            },
            {
                "question": "Find the theoretical limiting angular resolution (in arcsec) of a commercial 8-inch (diameter) optical telescope being used in the visible spectrum (at $\\lambda=5000 \\AA$). Answer in arcseconds to two significant figures.",
                "answer": "The angular resolution is $\\theta \\approx 1.22 \\lambda / D$.\n"
                          "$D = 8 \\text{ inches} = 0.2032 \\text{ m}$, $\\lambda = 5 \\times 10^{-7} \\text{ m}$.\n"
                          "$\\theta = 1.22 \\times 5 \\times 10^{-7} / 0.2032 = 3.0 \\times 10^{-6}$ rad\n"
                          "$= 3.0 \\times 10^{-6} \\times 206265 = \\boxed{0.49}$ arcseconds.\n"
                          "The answer is: 0.49."
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
        # \boxed{}
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
        final_pattern = r"Final answer:?\s*(?:The final answer is\s*)?\$?([^\$\n]+)\$?"
        final_matches = re.findall(final_pattern, prediction, re.IGNORECASE)
        if final_matches:
            pred_ans = final_matches[-1].strip().rstrip(".")
            return normalize_math_string(pred_ans) == normalize_math_string(gold_answer)

        # Fallback: So sánh số cuối cùng trong prediction với gold (cho đáp án dạng số)
        try:
            gold_num = float(gold_answer.replace(",", ""))
            nums = re.findall(r'-?[\d]+\.?[\d]*(?:e[+-]?\d+)?', prediction.replace(",", ""), re.IGNORECASE)
            if nums:
                pred_num = float(nums[-1])
                # Tolerance cho đáp án xấp xỉ (< 2% error)
                if gold_num != 0:
                    return abs(pred_num - gold_num) / abs(gold_num) < 0.02
                else:
                    return abs(pred_num - gold_num) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass

        return False
