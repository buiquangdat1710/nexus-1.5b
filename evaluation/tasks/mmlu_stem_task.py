import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask

class MMLUStemTask(BaseEvalTask):
    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="cais/mmlu", split="test", num_shots=num_shots)
        self.system_prompt = "You are an expert in STEM subjects. Read the multiple-choice question, explain your reasoning, and choose the correct answer."
        self.choices = ["A", "B", "C", "D"]
        
        # 18 subset STEM
        self.stem_subjects = [
            "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "electrical_engineering",
            "elementary_mathematics", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_mathematics", "high_school_physics",
            "high_school_statistics", "machine_learning"
        ]
        
        self.golden_shots = [
            {
                "question": "Find the domain of the expression \\frac{\\sqrt{x-2}}{\\sqrt{5-x}}.\nWhat of the following is the right choice? Explain your answer.\n(A) [-5,-2) \n(B) [2,5) \n(C) [-2,-5) \n(D) [5,2)", 
                "answer": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is (B). I hope it is correct."
            },
            {
                "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$\nWhat of the following is the right choice? Explain your answer.\n(A) 14 \n(B) 4 \n(C) 2 \n(D) 24", 
                "answer": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is (D). I hope it is correct."
            },
            {
                "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nWhat of the following is the right choice? Explain your answer.\n(A) 12 \n(B) 20 \n(C) 16 \n(D) 15", 
                "answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot 20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot 15\\cdot n=30n$ pounds of weight.\nEquating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is (C). I hope it is correct."
            },
            {
                "question": "If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}\nhas a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.\nWhat of the following is the right choice? Explain your answer.\n(A) $-\\frac{2}{3}$ \n(B) $\\frac{2}{3}$ \n(C) $\\frac{1}{3}$ \n(D) $\\frac{4}{9}$", 
                "answer": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is (A). I hope it is correct."
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        examples = []
        gold_answers = []
        
        for subset in self.stem_subjects:
            try:
                ds = load_dataset(self.dataset_name, subset, split=self.split)
                
                for item in ds:
                    q = item["question"] + "\nWhat of the following is the right choice? Explain your answer.\n"
                    for i, choice in enumerate(item["choices"]):
                        q += f"({self.choices[i]}) {choice}\n"
                    q = q.strip()
                        
                    examples.append({"question": q})
                    
                    gold_answers.append(self.choices[item["answer"]])
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

        specific_pattern = r"The final answer is \(([A-D])\)"
        matches = re.findall(specific_pattern, prediction)
        if matches:
            return matches[-1] == gold_answer
            
        # fallback 
        bracket_pattern = r"\(([A-D])\)"
        bracket_matches = re.findall(bracket_pattern, prediction)
        if bracket_matches:
            return bracket_matches[-1] == gold_answer
        
        fallback_pattern = r'\b([A-D])\b'
        fallback_matches = re.findall(fallback_pattern, prediction.upper())
        if fallback_matches:
            return fallback_matches[-1] == gold_answer
            
        return False