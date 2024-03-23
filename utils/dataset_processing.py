"""  dataset_processing.py
    Helper functions for processing the GQA dataset. 
    This script contains helper functions to download the GQA dataset, load and transform images,
    get the path to save tensor files, download the GQA dataset, and trim question files.
    
"""
import os
import json
from PIL import Image
from tqdm import tqdm
import shutil

import torch
from torchvision import transforms


def hash_mod(filename, prime):
    return hash(filename) % prime

def load_and_transform_image(image_path, image_size=(224, 224)):
    """
    Load and transform an image from the given image path.

    Args:
        image_path (str): The path to the image file.
        image_size (tuple, optional): The desired size of the transformed image. Defaults to (224, 224).

    Returns:
        torch.Tensor: The transformed image as a tensor.
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    return transform(image)

def get_tensor_path(base_path, filename, primes):
    """
    Get the path to save the tensor file for a given image filename.

    Args:
        base_path (str): The base path where the tensor file will be saved.
        filename (str): The name of the image file.
        primes (list): A list of prime numbers used to create subdirectories.

    Returns:
        str: The path to save the tensor file.

    """
    sub_dirs = [str(hash_mod(filename, prime)) for prime in primes]
    save_dir = os.path.join(base_path, *sub_dirs)
    tensor_path = os.path.join(save_dir, filename.replace('.jpg', '.pt'))
    return tensor_path

def download_gqa():
    files_to_download = {
        "sceneGraphs": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
        "questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
        "images": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    }
    # # Download each file
    # for file_name, file_url in files_to_download.items():
    #     print(f"Downloading {file_name}...")
    #     !wget "$file_url" -O "$path{file_name}.zip"
    #     print(f"Saved to {path}{file_name}.zip")

def trim_question_files(source_questions_path, trimmed_questions_path, fields_to_keep):
    """
    Trims the question files to keep only specified fields.

    Parameters:
    - source_questions_path: Path to the directory containing the original question files.
    - trimmed_questions_path: Path to the directory where trimmed files will be saved.
    - fields_to_keep: A set of strings representing the fields to keep in each question.

    The function does not return any value but saves trimmed question files to the specified directory.
    """
    # Create the trimmed questions directory if it doesn't exist
    os.makedirs(trimmed_questions_path, exist_ok=True)

    # List all the file paths in the source questions directory
    paths_to_all_train_questions = [os.path.join(source_questions_path, file)
                                    for file in os.listdir(source_questions_path)
                                    if os.path.isfile(os.path.join(source_questions_path, file))]

    # Process each file
    for file_path in tqdm(paths_to_all_train_questions, desc='Trimming Questions'):
        with open(file_path, 'r', encoding='UTF-8') as file:
            # Load the questions data
            data = json.load(file)

        # Trim the data
        trimmed_data = {questionId: {field: questionData[field] for field in fields_to_keep if field in questionData}
                        for questionId, questionData in data.items()}

        # Define the new file name
        trimmed_file_name = os.path.basename(file_path).replace('.json', '_trimmed.json')
        trimmed_file_path = os.path.join(trimmed_questions_path, trimmed_file_name)

        # Save the trimmed data to a new JSON file
        with open(trimmed_file_path, 'w', encoding='UTF-8') as file:
            json.dump(trimmed_data, file, indent=4)

    print(f"All files processed and saved in {trimmed_questions_path}.")

def save_tensors_in_hashed_dicts(image_base_path, tensor_destination_base_path, prime):
    """
    Processes images, converts them to tensors, and organizes them into dictionaries
    based on the hash value of their filenames. Each dictionary corresponds to a unique
    hash value and is saved to disk.

    Args:
        image_base_path (str): The path to the directory containing the original image files.
        tensor_destination_base_path (str): The path where the dictionaries of tensorized images will be saved.
        prime (int): The prime number used for hashing to determine dictionary keys.
    """
    # Initialize dictionaries to store tensors
    tensor_dicts = {i: {} for i in range(prime)}

    # Process each image file in the base path
    for image_filename in tqdm(os.listdir(image_base_path), desc="Processing Images"):
        image_path = os.path.join(image_base_path, image_filename)

        # Transform the image into a tensor
        image_tensor = load_and_transform_image(image_path)

        # Determine the dictionary key based on the hash of the filename
        dict_key = hash_mod(image_filename, prime)

        # Add the tensor to the appropriate dictionary
        tensor_dicts[dict_key][image_filename] = image_tensor

    # Save each dictionary to disk
    for dict_key, tensor_dict in tensor_dicts.items():
        save_path = os.path.join(tensor_destination_base_path, f'tensors_dict_{dict_key}.pt')
        torch.save(tensor_dict, save_path)
        print(f"Saved {len(tensor_dict)} tensors in '{save_path}'")



def copy_images_into_directories(image_base_path, destination_base_path, prime):
    """
    Copies image files from the given base path into multiple directories
    based on the hash value of their filenames and a prime number.

    Args:
        image_base_path (str): The path to the directory containing the original image files.
        destination_base_path (str): The base path where images will be copied into hashed directories.
        prime (int): A prime number used for hashing to determine the directory structure.
    """
    # Ensure the base directory for the copied images exists
    if not os.path.exists(destination_base_path):
        os.makedirs(destination_base_path, exist_ok=True)

    # Process each image file in the base path
    for image_filename in tqdm(os.listdir(image_base_path), desc="Copying Images"):
        # Determine the directory using the hash of the filename
        directory_name = str(hash_mod(image_filename, prime))
        destination_dir = os.path.join(destination_base_path, directory_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)

        # Copy the file
        source_path = os.path.join(image_base_path, image_filename)
        destination_path = os.path.join(destination_dir, image_filename)
        shutil.copy(source_path, destination_path)

    print(f"All images copied to {destination_base_path}.")


