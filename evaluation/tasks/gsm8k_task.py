import re
from datasets import load_dataset
from typing import List, Dict, Tuple
from evaluation.tasks.base_task import BaseEvalTask

class GSM8KTask(BaseEvalTask):
    def __init__(self, num_shots: int = 8):
        super().__init__(dataset_name="openai/gsm8k", split="test", num_shots=num_shots)
        self.system_prompt = "You are a helpful math assistant."
        
        self.golden_shots = [
            {
                "question": "In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?", 
                "answer": "In 2005, 60/2=30 kids came to the cookout.\nIn 2006, 30/3*2=20 kids came to the cookout.\nThe answer is 20"
            },
            {
                "question": "Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?", 
                "answer": "Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.\nThe total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.\nSo, $1900/2 = $950 is spent on her other monthly expenses.\nThe total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.\nHence, she saves $1900 - $1083 = $817 per month.\nThe answer is 817"
            },
            {
                "question": "If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?", 
                "answer": "The total ratio representing the slices of pizza that Buzz bought is 5+8=13\nIf he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices\nTwenty less the number of slices of pizza that the waiter ate is 48-20 = 28\nThe answer is 28"
            },
            {
                "question": "Jame gets a raise to $20 per hour and works 40 hours a week. His old job was $16 an hour for 25 hours per week. How much more money does he make per year in his new job than the old job if he works 52 weeks a year?", 
                "answer": "He makes 20*40=$800 per week\nHe used to make 16*25=$400 per week\nSo his raise was 800-400=$400 per week\nSo he makes 400*52=$20,800 per year more\nThe answer is 20800"
            },
            {
                "question": "Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?", 
                "answer": "Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats\nEach student will receive 80 / 20 = 4 sweet treats\nThe answer is 4"
            },
            {
                "question": "A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot's vehicles in all?", 
                "answer": "The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.\nThe lot has 24 - 8 = 16 cars for sale\nThere are 16 / 4 = 4 cars with a spare tire with 5 tires each.\nThe lot has 16 - 4 = 12 cars with 4 tires each.\nThus, the used car lot's vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.\nThe answer is 84"
            },
            {
                "question": "Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?", 
                "answer": "Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters\nAdding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes\nWhen she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts\nAdding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes\nSubtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing\nThe answer is 15"
            },
            {
                "question": "Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?", 
                "answer": "During 30 days Adam picked 4 * 30 = 120 apples.\nSo in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.\nThe answer is 350"
            }
        ]

    def load_data(self) -> Tuple[List[Dict], List[str]]:
        ds = load_dataset(self.dataset_name, "main", split=self.split)
        examples = [{"question": item["question"]} for item in ds]
        # GSM8K lưu đáp án sau chuỗi "#### "
        gold_answers = [item["answer"].split("####")[-1].strip().replace(",", "") for item in ds]
        return examples, gold_answers

    def build_prompt(self, problem: str, few_shot_examples: List[Dict]) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        for ex in few_shot_examples:
            prompt += f"<|im_start|>user\nQuestion: {ex['question']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\nLet's think step by step\n{ex['answer']}<|im_end|>\n"

        # question    
        prompt += f"<|im_start|>user\nQuestion: {problem}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\nLet's think step by step\n"
        return prompt

    def evaluate_correctness(self, prediction: str, gold_answer: str) -> bool:

        matches = re.findall(r"The answer is\s*([0-9,\.\-]+)", prediction, re.IGNORECASE)
        if matches:
            pred_ans = matches[-1].replace(",", "").strip()
            return pred_ans == gold_answer
            
        # fallback
        nums = re.findall(r'-?\d+\.?\d*', prediction.replace(",", ""))
        if nums:
            pred_ans = nums[-1]
            return pred_ans == gold_answer
            
        return False