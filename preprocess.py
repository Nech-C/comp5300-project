"""

"""
import argparse
import json
import random

def preprocess_gqa(input_file, output_file, prompt):
    """
    Preprocess the GQA dataset for training.
    
    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to save the processed data.
        
    """
    with open(input_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for question_id, question_info in data.items():
        image_name = question_info['imageId'] + '.jpg'

        trainable_question = {
            'id': question_id,
            'image': image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": question_info['question']
                },
                {
                    "from": "gpt",
                    "value": question_info['answer']
                }
            ],
        }
        processed_data.append(trainable_question)

    # Write the processed data back to the original file, effectively overwriting it
    with open(output_file, 'w', encoding="utf-8") as f:
        # Convert the list of dictionaries to a JSON string and write to the file
        json.dump(processed_data, f, indent=4)

def preprocess_vqa(input_file, output_file, prompt):
    ...

def preprocess_for_inference(input_file, output_file, prompt):
    """
    Preprocess the data for inference.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to save the processed data.
        prompt (str): The prompt to use for inference/training.
    """
    all_questions = []
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            q_id = item['id']
            image = item['image']
            text = item['conversations'][0]['value']

            question = {
                "question_id": q_id,
                "image": image,
                "text": prompt + text
            }

            all_questions.append(question)

    random.shuffle(all_questions)

    with open(output_file, "w", encoding="utf-8") as output_file:
        for question in all_questions:
            output_file.write(json.dumps(question) + "\n")

    print(f"Conversion completed. {len(all_questions)} questions written to the output file.")
    print("Output file:", all_questions)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for training.")
    parser.add_argument("--task_type", type=str, required=True,
                        help="The dataset to preprocess. Options: 'gqa', 'vqa'")
    parser.add_argument("--input_file", type=str, required=True,
                        help="The path to the input file.")
    parser.add_argument("--output_path ", type=str, required=True,
                        help="The path to save the processed data.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="The prompt to use for inference/training.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.task_type == "gqa":
        preprocess_gqa(args.input_file, args.output_file, args.prompt)
    elif args.task_type == "vqa":
        preprocess_vqa(args.input_file, args.output_file, args.prompt)
    elif args.task_type == "inference":
        preprocess_for_inference(args.input_file, args.output_file, args.prompt)
    else:
        print("Invalid task type. Please choose 'gqa', 'vqa', or 'inference'.")

if __name__ == "__main__":
    main()
