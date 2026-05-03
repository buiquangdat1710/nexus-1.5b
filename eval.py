import argparse
from tqdm import tqdm

from evaluation.evaluator import MathEvaluator

from evaluation.tasks.math_task import MathTask
from evaluation.tasks.gsm8k_task import GSM8KTask
from evaluation.tasks.mmlu_stem_task import MMLUStemTask
from evaluation.tasks.cmath_task import CMATHTask
from evaluation.tasks.gaokao_cloze_task import GaoKaoClozeTask
from evaluation.tasks.gaokao_qa_task import GaoKaoQATask

TASKS = {
    "math": MathTask,
    "gsm8k": GSM8KTask,
    "mmlu_stem": MMLUStemTask,
    "cmath": CMATHTask,
    "gaokao_cloze": GaoKaoClozeTask,
    "gaokao_qa": GaoKaoQATask
}

def parse_args():
    p = argparse.ArgumentParser(description="Nexus Evaluation Framework")
    p.add_argument("--model_path", type=str, required=True, help="Đường dẫn đến model")
    p.add_argument("--task", type=str, required=True, choices=TASKS.keys(), help="Tên Task cần đánh giá")
    p.add_argument("--tp", type=int, default=1, help="Tensor Parallel size (Số lượng GPU muốn dùng, mặc định 1)")
    p.add_argument("--temperature", type=float, default=0.0, help="Greedy decoding = 0.0")
    return p.parse_args()

def main():
    args = parse_args()
    
    TaskClass = TASKS[args.task]
    task = TaskClass()
    
    print(f"Bắt đầu đánh giá Task: {args.task.upper()} ({task.num_shots}-shot)")
    
    print("Đang tải dữ liệu test...")
    raw_examples, gold_answers = task.load_data()
    
    if not raw_examples:
        print("Không có dữ liệu để đánh giá. Vui lòng kiểm tra lại quá trình load dataset.")
        return

    # Golden Few-shots
    few_shot_examples = task.generate_few_shots()

    # Build Prompts
    prompts = [task.build_prompt(ex["question"], few_shot_examples) for ex in raw_examples]

    # Model Evaluator và Generate
    evaluator = MathEvaluator(model_path=args.model_path, tensor_parallel_size=args.tp)
    print(f"Đang sinh câu trả lời cho {len(prompts)} câu hỏi...")
    predictions = evaluator.generate_answers(prompts, temperature=args.temperature)

    # Evaluation
    print("Evaluating...")
    correct = 0
    for pred, gold in tqdm(zip(predictions, gold_answers), total=len(predictions), desc="Evaluating"):
        if task.evaluate_correctness(pred, gold):
            correct += 1

    accuracy = (correct / len(prompts)) * 100
    print("\n" + "="*50)
    print("BÁO CÁO BENCHMARK")
    print("="*50)
    print(f"Model    : {args.model_path}")
    print(f"Task     : {args.task.upper()} ({task.num_shots}-shot)")
    print(f"Accuracy : {accuracy:.2f}% ({correct}/{len(prompts)})")
    print("="*50)

if __name__ == "__main__":
    main()