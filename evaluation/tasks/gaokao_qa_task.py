import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask

class GaoKaoQATask(BaseEvalTask):
    def __init__(self, num_shots: int = 4):
        super().__init__(dataset_name="hails/agieval-gaokao-mathqa", split="test", num_shots=num_shots)

        self.system_prompt = "你是一个高考数学专家。仔细阅读下面的选择题，给出严谨的推导过程，并选出正确的选项 (A, B, C 或 D)."
        
        self.golden_shots = [
            {
                "question": "下列有关命题的说法正确的是( )\nA. 命题“若$ x^{2}=1 $, 则$ x=1 $”的否命题为：“若$ x^{2}=1 $, 则$ x\\neq 1 $” \nB. 命题“若$ x=y $, 则$ \\sin x=\\sin y $”的逆否命题为真命题 \nC. 命题“存在$ x\\in R $, 使得$ x^{2}+x+1 < 0 $”的否定是：“对任意$ x\\in R $, 均有$ x^{2}+x+1 < 0 $” \nD. “$ x=-1 $”是“$ x^{2}-5x-6=0 $”的必要不充分条件", 
                "answer": "命题“若$ x^{2}=1 $,则$ x=1 $”的否命题为“若$ x^{2}\\neq 1 $,则$ x\\neq 1 $”,故排除A;\n$\\because$命题“若$ x=y $,则$ \\sin x=\\sin y $”为真命题,故其逆否命题为真命题,B正确;\n命题“存在$ x\\in R $,使得$ x^{2}+x+1 < 0 $”的否定是:“对任意$ x\\in R $,均有$ x^{2}+x+1 \\geqslant 0 $”,故排除C;\n$\\because$“$ x^{2}-5x-6=0 $” $\\Leftrightarrow$ “$ x=-1 $或$ x=6 $”,$\\therefore$“$ x=-1 $”是“$ x^{2}-5x-6=0 $”的充分不必要条件,排除D;\n故选:B.\n推理结束。"
            },
            {
                "question": "已知函数$ f(x)=2x^{2}+mx-1 $,若对于任意$ x\\in[m,m+1] $,都有$ f(x)<0 $成立,则实数$ m $的取值范围是( )\nA. $ \\left(-\\sqrt{2},0\\right] $ \nB. $ \\left(-2,0\\right) $ \nC. $ \\left[-\\dfrac{\\sqrt{2}}{2},0\\right] $ \nD. $ \\left(-\\dfrac{\\sqrt{2}}{2},0\\right) $", 
                "answer": "由题意可得$\\begin{cases}f(m)=2m^{2}-1 < 0 \\\\ f(m+1)=2(m+1)^{2}+m(m+1)-1 < 0\\end{cases}$, \n求得$ -\\dfrac{\\sqrt{2}}{2} < m < 0 $,\n即实数$ m $的取值范围为$ \\left(-\\dfrac{\\sqrt{2}}{2},0\\right) $.\n故选:D.\n推理结束。"
            },
            {
                "question": "设$ i $是虚数单位,若复数$ a+\\dfrac{5i}{1-2i}(a\\in R) $是纯虚数,则$ a $等于( )\nA. -1 \nB. 1 \nC. 2 \nD. -2", 
                "answer": "$\\because a+\\dfrac{5i}{1-2i}=a+\\dfrac{5i(1+2i)}{(1-2i)(1+2i)}=a+\\dfrac{-10+5i}{5}=a-2+i$是纯虚数,\n$\\therefore a=2$.\n故选:C.\n推理结束。"
            },
            {
                "question": "已知集合$ A=\\{x|2\\leqslant x < 7\\} $, $ B=\\{x|3 < x < 10\\} $, $ C=\\{x|a-5 < x < a\\} $. 若非空集合$ C\\subseteq(A\\cup B) $,则$ a $的取值范围是( )\nA. $ 7\\leqslant a\\leqslant 10 $ \nB. $ 7\\leqslant a < 10 $ \nC. $ 8 < a < 10 $ \nD. $ 8\\leqslant a\\leqslant 10 $", 
                "answer": "$\\because$集合$ A=\\{x|2\\leqslant x < 7\\} $, $ B=\\{x|3 < x < 10\\} $,\n$\\therefore A\\cap B=\\{x|3 < x < 7\\} $,\n$ A\\cup B=\\{x|2\\leqslant x < 10\\} $,\n当$ C\\neq \\varnothing $时,要使$ C\\subseteq(A\\cup B) $,$\\begin{cases}a-5\\geqslant 2 \\\\ a\\leqslant 10\\end{cases}$,解得$ 7\\leqslant a\\leqslant 10 $;\n$\\therefore a $的取值范围是$ 7\\leqslant a\\leqslant 10 $.\n故选:A.\n推理结束。"
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        try: 
            ds = load_dataset(self.dataset_name, split=self.split)
            examples = []
            gold_answers = []
            
            idx_to_char = {0: "A", 1: "B", 2: "C", 3: "D"}
            
            for item in ds:
                q_text = item["query"]
                
                # bỏ prefix "问题：" ở đầu (nếu có)
                if q_text.startswith("问题：") or q_text.startswith("问题:"):
                    q_text = q_text[3:].strip()
                    
                examples.append({"question": q_text})
                
                gold_idx = int(item["gold"][0]) 
                gold_answers.append(idx_to_char.get(gold_idx, "A")) # fallback
                
            return examples, gold_answers
            
        except Exception as e:
            print(f"Lỗi khi load {self.dataset_name}: {e}. Vui lòng kiểm tra lại dataset.")
            return [], []

    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for ex in few_shot_examples:
            prompt += f"<|im_start|>user\n选择题: {ex['question']}<|im_end|>\n<|im_start|>assistant\n解:{ex['answer']}<|im_end|>\n"
        prompt += f"<|im_start|>user\n选择题: {problem}<|im_end|>\n<|im_start|>assistant\n解:"
        return prompt

    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:
        specific_pattern = r"故选\s*[:：]?\s*([A-D])"
        matches = re.findall(specific_pattern, prediction.upper())
        if matches:
            return matches[-1] == gold_answer
            
        fallback_pattern = r'\b([A-D])\b'
        fallback_matches = re.findall(fallback_pattern, prediction.upper())
        if fallback_matches:
            return fallback_matches[-1] == gold_answer
            
        return False