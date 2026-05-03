import re
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness

class MathTask(BaseEvalTask):
    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="EleutherAI/hendrycks_math", split="test", num_shots=num_shots)

        self.system_prompt = "You are a mathematics expert. Solve the problem step by step."
        
        # 7 subset
        self.subsets = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus"
        ]
        
        self.golden_shots = [
            {
                "question": "Find the domain of the expression \\frac{\\sqrt{x-2}}{\\sqrt{5-x}}.", 
                "answer": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nThe answer is: $[2,5)$."
            },
            {
                "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$", 
                "answer": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nThe answer is: $24$."
            },
            {
                "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?", 
                "answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot 20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot 15\\cdot n=30n$ pounds of weight.\nEquating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nThe answer is: $16$."
            },
            {
                "question": "If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x&=b.\n\\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.", 
                "answer": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nThe answer is: $-\\frac{2}{3}$."
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        examples = []
        gold_answers = []
        
        for subset in self.subsets:
            try:
                ds = load_dataset(self.dataset_name, subset, split=self.split)
                
                for item in ds:
                    examples.append({"question": item["problem"]})
                    gold_answers.append(item["solution"])
                    
            except Exception as e:
                print(f"Lỗi khi load subset {subset}: {e}")
                
        return examples, gold_answers

    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        for ex in few_shot_examples:
            prompt += f"<|im_start|>user\nProblem:\n{ex['question']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\nSolution:\n{ex['answer']}<|im_end|>\n"
            
        # question
        prompt += f"<|im_start|>user\nProblem:\n{problem}\n<|im_end|>\n"
        prompt += f"<|im_start|>assistant\nSolution:\n"
        
        return prompt

    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:

        # \boxed{}
        is_correct = check_correctness(prediction, gold_answer)
        if is_correct:
            return True
            
        # fallback
        import re
        fallback_pattern = r"The answer is:\s*\$?([^\$\n]+)\$?"
        matches = re.findall(fallback_pattern, prediction)
        
        if matches:
            pred_ans = matches[-1].strip()
            if pred_ans.endswith("."):
                pred_ans = pred_ans[:-1]
                
            from evaluation.metrics import normalize_math_string
            return normalize_math_string(pred_ans) == normalize_math_string(gold_answer)
            
        return False