from os.path import abspath, dirname, split, join
import random
import re

import argparse
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
import evaluate
import nltk
from torch.utils.data import DataLoader
import torch
import gc

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

from peft import (
    get_peft_config,
    get_peft_model,
    PeftModel, #
    PeftConfig, #
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)

nltk.download('punkt')

device = ("cuda" if torch.cuda.is_available() else "cpu")

ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--dataset-name-or-path-1', type=str, default="./data/meeting_summary_ami.json")
    parser.add_argument('--dataset-name-or-path-2', type=str, default="./data/meeting_summary_icsi.json")
    parser.add_argument('--model-name-or-path', type=str, default="bigscience/bloomz-3b")
    parser.add_argument('--peft-model-name', type=str, default="bloomz-3b_PROMPT_TUNING_CAUSAL_LM_epoch20_202304182334")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--task-prompt', type=str, default=" TL;DR: ")
    parser.add_argument('--num-virtual-tokens', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=924)
    parser.add_argument('--uid', type=str, default="27-ES2003b")
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if device == "cuda" else None

def dataset_split(dataset_ori: Dataset) -> DatasetDict:
    dataset_train = dataset_ori.filter(lambda x: x["split"] == "train")
    dataset_valid = dataset_ori.filter(lambda x: x["split"] == "valid")
    dataset_test = dataset_ori.filter(lambda x: x["split"] == "test")

    dataset_split = DatasetDict({
        "train": dataset_train,
        "valid": dataset_valid,
        "test": dataset_test
    })
    return dataset_split

def add_task_prompt(example):
    example["source"] = example["text"] + task_prompt
    return example

def tokenize(batch):
    return tokenizer(
        batch["source"], 
        padding='max_length', 
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
        )

def calculate_model_size(model, model_name_or_path):
    num_params = sum(t.numel() for t in model.parameters())
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print(f"{model_name_or_path} size: {num_params/1000**2:.1f}M parameters")
    print(f"model stored size: {size_all_mb:.3f}MB")


def predict_one_sample(model, task_prompt, text=None, row_of_dataset=None, max_new_tokens=48, repetition_penalty=1.05):
    if row_of_dataset:
        text = row_of_dataset["source"][0]
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    model, inputs = accelerator.prepare(model, inputs)
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}       
            outputs = accelerator.unwrap_model(model).generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, repetition_penalty=repetition_penalty)
        
        outputs = accelerator.gather(outputs).cpu().numpy()
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_start = re.search(task_prompt, output_text).span()[1]
        # find the last period "."
        try:
            output_end = re.search(task_prompt+r'(.*?)(.*)\.', output_text).span()[1]
        except:
            # if there's no period after task_prompt
            output_end = len(output_text)

        prediction = output_text[output_start:output_end]
        reference = row_of_dataset['summary'][0]
        print(f"\nuid: {row_of_dataset['uid'][0]}")
        print(f"\nPredicted summary:\n{prediction}")
        print(f"\nGround truth summary:\n{reference}")
        
        del outputs, inputs
        gc.collect()

    return [prediction], [reference]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    dataset_name_or_path_1 = args.dataset_name_or_path_1
    dataset_name_or_path_2 = args.dataset_name_or_path_2
    model_name_or_path = args.model_name_or_path
    peft_model_name = args.peft_model_name
    output_dir = args.output_dir
    task_prompt = args.task_prompt
    num_virtual_tokens = args.num_virtual_tokens
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_length = args.max_length
    uid = args.uid

    set_seed(seed)

    ## 1. Data preprocess
    ## 1-1. Import the preprocessed data, and then combine them
    dataset_ami_ori = load_dataset("json", data_files=dataset_name_or_path_1, split="train")
    dataset_icsi_ori = load_dataset("json", data_files=dataset_name_or_path_2, split="train")

    meeting_dataset_ori = concatenate_datasets([dataset_ami_ori, dataset_icsi_ori])

    ## 1-2. Train / validation / test split based on featrue "split"
    # meeting_dataset = dataset_split(meeting_dataset_ori)
    # do not filter the data split now


    ## 1-3. Add task-specific prompt to concatenate text and summary at train and valid split
    processed_data = meeting_dataset_ori.map(add_task_prompt)
    

    ## 1-4. Import model tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ## 1-5. Filter out the example with too long `source`
    tknz_data = processed_data.map(tokenize, batched=True, remove_columns=["uid", "id", "text", "summary", "split", "source"])
    filterd_data = tknz_data.filter(lambda x: len(x["input_ids"]) <= max_length)

    # creating model
    peft_model_id = f"{output_dir}/{peft_model_name}"

    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
        )

    model = PeftModel.from_pretrained(
        model, 
        peft_model_id, 
        torch_dtype=torch.float16,
        device_map="auto")
    
    model.print_trainable_parameters()
    calculate_model_size(model, model_name_or_path)

    accelerator = Accelerator()

    
    # inference
    ROW_OF_DATASET = processed_data.filter(lambda x: x["uid"] == uid)
    predictions, references = predict_one_sample(model, task_prompt, row_of_dataset=ROW_OF_DATASET, 
                                                 max_new_tokens=50, repetition_penalty=1.06)
    predictions, references = postprocess_text(predictions, references)

    rouge_score = evaluate.load("rouge")
    result = rouge_score.compute(predictions=predictions, references=references)
    print(result)
    
    