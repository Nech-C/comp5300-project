# comp5300-project

## Installation (essential)

### Install pytorch with cuda
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install git-lfs

```bash
git lfs install
```

### Install other pacages

### Clone LLaVA Repo:
```Bash
git clone https://github.com/haotian-liu/LLaVA
```
### Clone this Repo:
```Bash
git clone https://github.com/Nech-C/comp5300-project
```
### Download the LLaVA 1.5 weights:
```Bash
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```
## Preparation (Training)

### Using the GQA dataset:
Visit https://cs.stanford.edu/people/dorarad/gqa/download.html and download the image files.  
Next, download the question/answer json files that have been converted to the trainable format:
```Bash
git clone https://huggingface.co/datasets/Nech-C/gqa-lora
```
The repository above only contain the processed version of a portion of all training data. If you wish to convert other question json files into trainable format, run the following command:

```bash
python preprocess.py --task_type gqa --input_file path/to/file --output_path path/to/output
```

### Using your own dataset:
you have to format your data to the following format:
```
[
    {
        "id": "id of the question",
        "image": "file_name.jpg",
        "conversations" : [
            {
                "from": "human",
                "value": "insert a question about the image"
            },
            {
                "from": "gpt",
                "value": "formatted_answers"
            }
        ]
    },
]

```

Make sure that you mantain the values for the two "from" properties as "human" and "gpt". "id", "image", and "value" are user defined properties.


## Preparation (Inference)
### Using a trainable json format:
We again need to format the data into a different format. To convert a .json file that has been converted to the trainable format, run the following:

```bash
python preprocess.py --task_type inference --input_file path/to/file --output_path path/to/output
```

### Using some other formats:
You will have to convert the file into the following format:
```
[
    {
        "question_id": "id",
        "image": "image.jpg",
        "text": "question"
    }
]
```



## Finetune using LoRA/QLoRA
run the following script:
```
deepspeed ./finetune_LLaVA/llava/train/train_mem.py \
--deepspeed ./finetune_LLaVA/scripts/zero2.json \
--lora_enable True \
--lora_r 126\
--lora_alpha 256 \
--mm_projector_lr 2e-5 \
--model_name_or_path ./llava-v1.5-7b \
--version llava_llama_2 \
--data_path ./gqa-lora/train_balanced_questions/train_balanced_questions.json \
--validation_data_path ./gqa-lora/testdev_balanced_questions/testdev_balanced_questions.json \
--image_folder ./images \
--vision_tower openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./checkpoints/llava-2-7b-chat-task-qlora \
--num_train_epochs 1 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "epoch" \
--save_strategy "steps" \
--save_steps 500 \
--save_total_limit 1 \
--learning_rate 4e-5 \
--weight_decay 0. \
--warmup_ratio 0.02 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 2 \
--lazy_preprocess True \
--report_to wandb
```

Feel free to change the hyperparameters like the lr, lora_r, etc. Also, bf16 and tf32 are only supported by Nvidia GPUs with Ampere architecture or newer. Please be sure to INCLUDE "llava", "checkpoints", "task", and "qlora/lora" so that the scripts works properly.

If you want to train with QLoRA, add the following line to the above command:
```
--bits 4 \
```

## Inference (for a single sample)
Run the following script:
```
python /content/finetune_LLaVA/llava/eval/run_llava.py --model-path /path/to/output/from/finetuning \
--model-base ./llava-v1.5-7b \
--image-file path/to/image.jpg \
--query "What are they doing? answer with less than 5 words."
```

Note: model-base refers to the base LLaVA model, and the model-path refers to the LoRA/QLoRA weights.



## Evaluation
### 1. Getting outputs for evaluation
Run the following script to generate the answers for the questions:
```
python /content/finetune_LLaVA/llava/eval/model_vqa.py \
--model-path ./path/to/the/lora/weights \
--model-base ./llava-v1.5-7b \
--image-folder ./images-lora \
--question-file ./converted_data.jsonl \
--answers-file ./path/to/output_file.jsonl
```

Note: The evaluation cannot run in batch, so choose the number of examples and hardware wisely.

### 2. Evaludate the answers
Run the following script:

```Bash
python eval.py --answers ./path/to/output_file.jsonl --ground_truth ./path/to/trainable.json
```

The script will return an accuracy for the answers. For the script to run correctly, make sure that the question ids match up. 