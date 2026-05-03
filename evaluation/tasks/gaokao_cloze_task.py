import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask
from evaluation.metrics import check_correctness

class GaoKaoClozeTask(BaseEvalTask):
    def __init__(self, num_shots: int = 5):
        super().__init__(dataset_name="hails/agieval-gaokao-mathcloze", split="test", num_shots=num_shots)
        
        self.system_prompt = "你是一个高考数学专家。请解答这道填空题，并给出严谨的推导过程。最终的填空答案请放在 \\boxed{} 中。"
        
        self.golden_shots = [
            {
                "question": "设数列 $\\left\\{ a_{n} \\right\\}$ 的前 $n$ 项和为 $S_{n}$，且 $a_{1}=-1$，$a_{n+1}=S_{n+1} S_{n}$，则 $S_{n}=$ (\\quad).", 
                "answer": "让我们写出这个数列的前n项和：\n$S_n = a_1 + a_2 + ... + a_n$\n$S_n = -1 + (S_2 S_1) + (S_3 S_2) + ... + (S_{n+1} S_n)$\n$S_n = -1 + S_n (S_{n+1} - S_1) = -1 - S_n - S_n S_{n+1} = -1 - S_n$\n$S_n (1 - S_{n+1}) = -1 - S_n$\n$S_n = -\\frac{1}{1 - S_{n+1}}$\n因为这个数列后面的所有项都是0，我们可以看到对于所有$n \\geq 1$，\n$S_{n+1} = 0$。因此，我们有：\n$S_n = -\\frac{1}{1 - S_{n+1}} = -\\frac{1}{1 - 0} = -1$\n这个数列前n项和的公式是$S_n = -\\frac{1}{n}$。\n答案是 $-\\frac{1}{n}$ \\boxed{-\\frac{1}{n}}"
            },
            {
                "question": "若 $\\left(x+\\frac{1}{x}\\right)^{n}$ 的展开式中第 3 项与第 7 项的二项式系数相等，则该展 开式中 $\\frac{1}{x^{2}}$ 的系数为 (\\quad).", 
                "answer": "由题意可得，$C_{n}^{2} = C_{n}^{6}$\n\\therefore n=8\n展开式的通项 $T_{r+1} = C_{8}^{r} x^{8-r} \\left(\\frac{1}{x}\\right)^{r} = C_{8}^{r} x^{8-2r}$\n令 $8-2r = -2$ 可得 $r=5$\n此时系数为 $C_{8}^{5} = 56$\n答案是 56 \\boxed{56}"
            },
            {
                "question": "函数 $\\mathrm{f}(\\mathrm{x})=\\sin (\\mathrm{x}+2 \\phi)-2 \\sin \\phi \\cos (\\mathrm{x}+\\phi)$ 的最大值为 (\\quad).", 
                "answer": "函数 $f(x) = \\sin(x+2\\phi) - 2\\sin\\phi \\cos(x+\\phi) = \\sin[(x+\\phi)+\\phi] - 2\\sin\\phi \\cos(x+\\phi)$\n$= \\sin(x+\\phi)\\cos\\phi + \\cos(x+\\phi)\\sin\\phi - 2\\sin\\phi \\cos(x+\\phi) = \\sin(x+\\phi)\\cos\\phi - \\cos(x+\\phi)\\sin\\phi$\n$= \\sin[(x+\\phi)-\\phi] = \\sin x$\n故函数 $f(x)$ 的最大值为 1\n答案是 1 \\boxed{1}"
            },
            {
                "question": "已知向量 $\\vec{a}=(3,1)$，$\\vec{b}=(1,0)$，$\\vec{c}=\\vec{a}+k \\vec{b}$。若 $\\vec{a} \\perp \\vec{c}$，则 $k=$ (\\quad).", 
                "answer": "因为 $\\vec{a}=(3,1)$，$\\vec{b}=(1,0)$，所以 $\\vec{c} = \\vec{a} + k\\vec{b} = (3+k, 1)$。\n因为 $\\vec{a} \\perp \\vec{c}$，所以 $\\vec{a} \\cdot \\vec{c} = 3(3+k) + 1 \\times 1 = 0$，解得 $k=-\\frac{10}{3}$\n答案是 $-\\frac{10}{3}$ \\boxed{-\\frac{10}{3}}"
            },
            {
                "question": "设向量 $\\vec{a}$，$\\vec{b}$ 不平行，向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行，则实数 $\\lambda=$ (\\quad).", 
                "answer": "因为向量 $\\vec{a}$，$\\vec{b}$ 不平行，向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行，\n所以 $\\lambda \\vec{a}+\\vec{b} = t(\\vec{a}+2 \\vec{b}) = t \\vec{a} + 2t \\vec{b}$\n所以 $\\left\\{ \\begin{array}{l} \\lambda = t \\\\ 1 = 2t \\end{array} \\right.$\n解得实数 $\\lambda = \\frac{1}{2}$。\n答案是 $\\frac{1}{2}$ \\boxed{\\frac{1}{2}}"
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        try: 
            ds = load_dataset(self.dataset_name, split=self.split)
            return [{"question": item["query"]} for item in ds], [item["answer"] for item in ds]
        except Exception as e:
            print(f"Lỗi khi load {self.dataset_name}: {e}. Vui lòng kiểm tra lại dataset.")
            return [], []

    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for ex in few_shot_examples:
            prompt += f"<|im_start|>user\n问题：\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n解析：\n{ex['answer']}<|im_end|>\n"
        prompt += f"<|im_start|>user\n问题：\n{problem}<|im_end|>\n<|im_start|>assistant\n解析：\n"
        return prompt

    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:
        # \boxed{}
        is_correct = check_correctness(prediction, gold_answer)
        if is_correct:
            return True
            
        # fallback
        fallback_pattern = r"答案是\s*([^\n\r]+)"
        matches = re.findall(fallback_pattern, prediction)
        
        if matches:
            pred_ans = matches[-1].strip()
            clean_pred = pred_ans.replace(" ", "").replace("$", "")
            clean_gold = gold_answer.replace(" ", "").replace("$", "")
            return clean_pred == clean_gold
            
        return False