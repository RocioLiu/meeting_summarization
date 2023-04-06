from os.path import abspath, dirname, split, join, isfile
import random


import argparse
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
import torch

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
    PeftModel,
    PeftConfig,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)

device = ("cuda" if torch.cuda.is_available() else "cpu")

ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--dataset-name-or-path-1', type=str, default="./data/meeting_summary_ami.json")
    parser.add_argument('--dataset-name-or-path-2', type=str, default="./data/meeting_summary_icsi.json")
    parser.add_argument('--model-name-or-path', type=str, default="bigscience/bloomz-560m")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--task-prompt', type=str, default=" TL;DR: ")
    parser.add_argument('--num-virtual-tokens', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=1948)
    # parser.add_argument('--lr', type=int, default=3e-5)
    # parser.add_argument('--weight-decay', type=int, default=0.02)
    # parser.add_argument('--scheduler-name', type=str, default="linear")
    # parser.add_argument('--num-warmup-steps', type=int, default=0)
    # parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    # parser.add_argument('--print-train-every-n-steps', type=int, default=10)
    # parser.add_argument('--eval-steps', type=int, default=200)

    # # to add weight and bias, we set
    # parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
    #                     help='if toggled, this experiment will be tracked with Weights and Biases')
    # parser.add_argument('--wandb-project-name', type=str, default="meeting-summarization",
    #                     help="the wandb's project name")
    # parser.add_argument('--wandb-entity', type=str, default=None,
    #                     help="the entity (team) of wandb's project")
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    # Set seed for everything
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

def add_task_prompt_test(example):
    example["source"] = example["text"] + task_prompt
    return example

def tokenize(batch):
    return tokenizer(
        batch["source"], 
        padding=True, 
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
#         return_overflowing_tokens=True,
#         return_length=True,
        )


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    dataset_name_or_path_1 = args.dataset_name_or_path_1
    dataset_name_or_path_2 = args.dataset_name_or_path_2
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    task_prompt = args.task_prompt
    num_virtual_tokens = args.num_virtual_tokens
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_length = args.max_length

    set_seed(seed)

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text="TL;DR: ", ##
        tokenizer_name_or_path=model_name_or_path,
        )

    ## 1. Data preprocess
    ## 1-1. Import the preprocessed data, and then combine them
    dataset_ami_ori = load_dataset("json", data_files=dataset_name_or_path_1, split="train")
    dataset_icsi_ori = load_dataset("json", data_files=dataset_name_or_path_2, split="train")

    meeting_dataset_ori = concatenate_datasets([dataset_ami_ori, dataset_icsi_ori])

    ## 1-2. Train / validation / test split based on featrue "split"
    meeting_dataset = dataset_split(meeting_dataset_ori)

    ## 1-3. Add task-specific prompt to concatenate text and summary at train and valid split
    processed_test = meeting_dataset["test"].map(add_task_prompt_test)

    ## 1-4. Import model tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tknz_test = processed_test.map(tokenize, batched=True,
                                      remove_columns=["id", "text", "summary", "split", "source"])

    ## 1-5. Filter out the example with too long `source`
    filterd_test = tknz_test.filter(lambda x: len(x["input_ids"]) <= max_length)


    # creating model
    from peft import PeftModel, PeftConfig

    peft_model_id = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}"
    # peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)

    i = 4
    inputs = tokenizer(processed_test[i]["source"], return_tensors="pt")
    print(processed_test[i]["source"])
    print(processed_test[i]["summary"])

    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50, eos_token_id=3
        )
        # print(outputs)
        # print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))