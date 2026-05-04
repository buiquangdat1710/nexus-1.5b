import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness, normalize_math_string


class OlympiadBenchTask(BaseEvalTask):
    """
    OlympiadBench (Hothan/OlympiadBench) - text-only English math subsets.
    Subsets: OE_TO_maths_en_COMP (674) + TP_TO_maths_en_COMP (503).
    """

    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="Hothan/OlympiadBench", split="train", num_shots=num_shots)
        self.system_prompt = (
            "You are an expert in mathematical olympiad problems. "
            "Solve the following problem step by step with rigorous reasoning. "
            "Put your final answer in \\boxed{}."
        )
        self.subsets = ["OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"]
        self.golden_shots = [
            {
                "question": "Find all positive integers $n$ such that $n^2 + 1$ divides $n! + 1$.",
                "answer": "Checking small values: $n=1$: $2|2$. For $n \\geq 2$, $n^2+1 > n!+1$ fails.\nThe only solution is $\\boxed{1}$.\nThe answer is: 1."
            },
            {
                "question": "Determine the maximum value of $\\sin(x) + \\sin(y) + \\sin(z)$ where $x+y+z=\\pi$ and $x,y,z \\geq 0$.",
                "answer": "By Jensen's inequality on the concave function $\\sin$:\n$\\sin x + \\sin y + \\sin z \\leq 3\\sin\\frac{x+y+z}{3} = 3\\sin\\frac{\\pi}{3} = \\frac{3\\sqrt{3}}{2}$.\nEquality when $x=y=z=\\pi/3$. Maximum is $\\boxed{\\frac{3\\sqrt{3}}{2}}$."
            },
            {
                "question": "How many ways can you tile a $2 \\times 10$ rectangle using $1 \\times 2$ dominoes?",
                "answer": "Let $f(n)$ be the number of ways to tile a $2 \\times n$ rectangle.\n$f(1)=1, f(2)=2$, and $f(n)=f(n-1)+f(n-2)$ (Fibonacci recurrence).\n$f(10) = \\boxed{89}$.\nThe answer is: 89."
            },
            {
                "question": "Let $p$ be a prime. Prove that $1^{p-1}+2^{p-1}+\\cdots+(p-1)^{p-1} \\equiv -1 \\pmod{p}$.",
                "answer": "By Fermat's Little Theorem, $a^{p-1} \\equiv 1 \\pmod{p}$ for $\\gcd(a,p)=1$.\nSo the sum $\\equiv (p-1) \\cdot 1 = p-1 \\equiv -1 \\pmod{p}$. $\\boxed{\\text{Proved}}$."
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        examples, gold_answers = [], []
        for subset in self.subsets:
            try:
                ds = load_dataset(self.dataset_name, subset, split=self.split)
                for item in ds:
                    question = item["question"]
                    context = item.get("context", "")
                    if context and context.strip():
                        question = context.strip() + "\n\n" + question
                    final_ans_list = item.get("final_answer", [])
                    if final_ans_list and len(final_ans_list) > 0:
                        examples.append({"question": question})
                        gold_answers.append(final_ans_list[0])
            except Exception as e:
                print(f"Lỗi khi load subset {subset}: {e}")
        return examples, gold_answers

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
        # Fallback: "The answer is: ..."
        for pattern in [r"The answer is:?\s*\$?([^\$\n]+)\$?", r"[Ff]inal [Aa]nswer:?\s*\$?([^\$\n]+)\$?"]:
            matches = re.findall(pattern, prediction, re.IGNORECASE)
            if matches:
                pred_ans = matches[-1].strip().rstrip(".")
                if normalize_math_string(pred_ans) == normalize_math_string(gold_answer):
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
