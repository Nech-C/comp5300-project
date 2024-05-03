import json
import argparse

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_answers(answers: list, ground_truth: dict):
    correct = 0
    total = len(answers)

    for answer in answers:
        question_id = answer['question_id']
        answer_text = answer['text'].lower()
        correct_answer = ground_truth[question_id]["conversations"][1]["value"].lower()
        if correct_answer in answer_text or answer_text in correct_answer or correct_answer == answer_text:
            correct += 1

    return correct / total

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model's answers.")
    parser.add_argument("--answers", type=str, help="The path to the answers file.")
    parser.add_argument("--ground_truth", type=str, help="The path to the ground truth file.")
    return parser.parse_args()

def main():
    args = parse_args()
    answers = read_jsonl(args.answers)
    ground_truth = read_jsonl(args.ground_truth)
    accuracy = evaluate_answers(answers, ground_truth)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
