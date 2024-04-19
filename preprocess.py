"""

"""
import argparse
import json

def preprocess_gqa(input_file, output_file):
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

def preprocess_vqa(input_file, output_file):
    ...

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for training.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to preprocess. Options: 'gqa', 'vqa'")
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input file.")
    parser.add_argument("--output_path ", type=str, required=True, help="The path to save the processed data.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.dataset == "gqa":
        preprocess_gqa(args.input_file, args.output_file)
    elif args.dataset == "vqa":
        preprocess_vqa(args.input_file, args.output_file)
    else:
        print("Invalid dataset. Please choose 'gqa' or 'vqa'.")
        
if __name__ == "__main__":
    main()
