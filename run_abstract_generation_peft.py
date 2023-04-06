import math
import json
import gc
import os
from os.path import abspath, dirname, split, join
import random
import re

from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
import argparse
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, load_dataset
from datetime import datetime
from distutils.util import strtobool
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
import wandb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
    # pipeline
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

# class CFG:
#     seed = 42
#     dataset_name_or_path = "gfissore/arxiv-abstracts-2021"
#     model_name_or_path = "bigscience/bloomz-560m"
#     output_dir = "/kaggle/working/bloom-560m-fintuned-abstract"
#     num_virtual_tokens = 128
#     num_epochs = 1
#     batch_size = 8
#     max_length = 256
#     lr = 3e-5
#     weight_decay = 0.02
#     scheduler_name = "linear"
#     num_warmup_steps = 50
#     gradient_accumulation_steps = 8
#     eval_steps = 1000

# args = CFG()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--dataset-name-or-path', type=str, default="gfissore/arxiv-abstracts-2021")
    parser.add_argument('--model-name-or-path', type=str, default="bigscience/bloomz-560m")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--num-virtual-tokens', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--lr', type=int, default=3e-5)
    parser.add_argument('--weight-decay', type=int, default=0.02)
    parser.add_argument('--scheduler-name', type=str, default="linear")
    parser.add_argument('--num-warmup-steps', type=int, default=50)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--eval-steps', type=int, default=1000)

    # to add weight and bias, we set
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="Abstract-generation",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    # Set seed for everything
    if seed is not None:
        random.seed(seed)
    #     np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if device == "cuda" else None

def add_year_column(example):
    example["year"] = extract_year(example["id"])
    return example

def extract_year(paper_id):
    if paper_id.find("/") >= 0:
        year_month_num = paper_id.split("/")[1]
        year_end = int(year_month_num[:2])
        if year_end > 90:
            year = 1900 + year_end
        else:
            year = 2000 + year_end
    else:
        year_end = int(paper_id.split(".")[0][:2])
        year = 2000 + year_end
    return year

def tokenize(batch):
    return tokenizer(
        batch["abstract"], 
        padding=True, 
        truncation=True,
        max_length=max_length, 
#         return_overflowing_tokens=True,
#         return_length=True,
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


def eval_fn(data_loader):
    model.eval()
    losses = []
    for step, batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            
        del batch
        gc.collect()
        torch.cuda.empty_cache()    
            
        losses.append(accelerator.gather(torch.unsqueeze(loss, -1))) 
        
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
        
    return loss.item(), perplexity.item()




if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    dataset_name_or_path = args.dataset_name_or_path
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    num_virtual_tokens = args.num_virtual_tokens
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_length = args.max_length
    lr = args.lr
    weight_decay = args.weight_decay
    scheduler_name = args.scheduler_name
    num_warmup_steps = args.num_warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    eval_steps = args.eval_steps
    track = args.track
    wandb_project_name = args.wandb_project_name
    wandb_entity=args.wandb_entity

    set_seed(seed)

    peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=num_virtual_tokens,
    prompt_tuning_init_text="Generate the following text with the starting sentence: ", ##
    tokenizer_name_or_path=model_name_or_path,
    )

    # output_dir = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}"
    peft_model_id = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}"
    run_name = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}_{datetime.now().strftime('%Y%m%d%H%M')}"

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    ## Experiment tracking with weights & Bias
    if track:
        run = wandb.init(project=wandb_project_name,
                         entity=wandb_entity,
                         name=run_name,
                         group=model_name_or_path.split("/")[-1],
                         config=vars(args),
                         sync_tensorboard=True,
                         save_code=True,
                         job_type="train")

    ## 1. Import and preprocess data
    my_dataset = load_dataset(dataset_name_or_path, split="train")
    my_dataset = my_dataset.remove_columns(['submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no', 'versions'])
    my_dataset = my_dataset.map(add_year_column)
    filtered_data = my_dataset.filter(lambda example: (example["year"] >= 2021) and ("cs" in example["categories"][0]))
    
    ## 2. Train / validation / test split
    ds = filtered_data.train_test_split(test_size=0.1, shuffle=True, seed=42)

    ## 3. Import model checkpoint and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", truncatopm_side="right")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ## 4. Data Preprocessing
    tknz_ds = ds.map(
        tokenize,
        batched=True,
        remove_columns = ['id', 'categories', 'abstract', 'year']
        )
    
    ## 5. Create DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(tknz_ds["train"], 
                              collate_fn=data_collator,
                              batch_size=batch_size, 
                              shuffle=True)
    eval_dataloader = DataLoader(tknz_ds["test"], 
                                collate_fn=data_collator,
                                batch_size=batch_size)
    
    ## 6. Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16
#     load_in_8bit=True, 
#     device_map='auto',
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    calculate_model_size(model, model_name_or_path)

    ## 7. Training hyperparameters
    # optimizer and lr scheduler
    # no_decay = ["bias", "ln_1.weight", "ln_2.weight"]
    no_decay = ["bias", "layernorm.weight", "layernorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    ## 8. Training loop
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    accelerator = Accelerator()
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    gc.collect()
    torch.cuda.empty_cache()

    completed_steps = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        model.train()
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.detach().float()
            
            del batch
            gc.collect()
            torch.cuda.empty_cache()    

            if step % 100 == 0:
                accelerator.print(f"steps: [{epoch+1}]{step}/{len(train_dataloader)} |",
                                f"updated_steps: {completed_steps} |",
                                f"lr: {lr_scheduler.get_last_lr()[0]:.7f} |",
    #                               f"loss/train: {loss.item() * gradient_accumulation_steps:.4f}",
                                f"loss/train: {loss.item():.4f}"
                                )
            if args.track:
                wandb.log({f"loss": loss.item(),
                        f"lr": lr_scheduler.get_last_lr()[0]})
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                
            if (step % (eval_steps * gradient_accumulation_steps)) == 0 or (step == (len(train_dataloader)-1)):
                eval_loss, eval_ppl = eval_fn(eval_dataloader)
                accelerator.print(f"loss/eval: {eval_loss:.4f} | perplexity: {eval_ppl:.4f}")
                model.train()
                
        avg_train_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(avg_train_loss)
        print(f"epoch {epoch+1}result:\n avg_train_loss={avg_train_loss:.4f} | train_ppl={train_ppl:.4f} | "
            f"avg_eval_loss={eval_loss:.4f} | eval_ppl={eval_ppl:.4f}")
        if args.track:
                wandb.log({f"epoch": epoch+1, 
                        f"avg_train_loss": avg_train_loss,
                        f"train_perplexity": train_ppl,
                        f"avg_eval_loss": eval_loss,
                        f"eval_perplexity": eval_ppl,
                        })
    #     model.save_pretrained(os.path.join(output_dir, peft_model_id))
    #     model.save_pretrained(f"{peft_model_id}_2")
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(peft_model_id, save_function=accelerator.save)
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(peft_model_id)

    if args.track:
        run.finish()



    






