import gc
import os
import random
import re
from os.path import abspath, dirname, split, join, isfile

from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
import argparse
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datetime import datetime
from distutils.util import strtobool
import evaluate
import nltk
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

from peft import (
    get_peft_config,
    get_peft_model,
    PeftModel,
    PeftConfig,
    LoraConfig,
    prepare_model_for_int8_training,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)

nltk.download('punkt')


ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--dataset-name-or-path-1', type=str,
                        default="./data/meeting_summary_ami.json")
    parser.add_argument('--dataset-name-or-path-2', type=str,
                        default="./data/meeting_summary_icsi.json")
    parser.add_argument('--model-name-or-path', type=str,
                        default="bigscience/bloomz-3b")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--task-prompt', type=str, default=" TL;DR: ")
    parser.add_argument('--num-virtual-tokens', type=int, default=100)
    parser.add_argument('--prompt-init-text', type=str,
                        default="Summarize the text: ")
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=int, default=0.05)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    # while using prompt-tuning, remember to deduct the num_virtual_tokens
    parser.add_argument('--max-length', type=int, default=924)
    parser.add_argument('--lr', type=int, default=3e-5)
    parser.add_argument('--weight-decay', type=int, default=0.02)
    parser.add_argument('--scheduler-name', type=str, default="cosine")
    parser.add_argument('--num-warmup-steps', type=int, default=2000)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--print-train-every-n-steps', type=int, default=25)
    parser.add_argument('--eval-steps', type=int, default=100)

    # to add weight and bias, we set
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="meeting-summarization",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

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
    example["source"] = example["text"] + task_prompt + example["summary"]
    return example


def add_task_prompt_test(example):
    example["source"] = example["text"] + task_prompt
    return example


def tokenize(batch):
    return tokenizer(
        batch["source"],
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )


def tokenize_no_maxlen(batch):
    return tokenizer(
        batch["source"]
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
    for step, batch in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                generated_tokens = accelerator.unwrap_model(model).generate(**batch,
                                                                            max_new_tokens=50,
                                                                            eos_token_id=tokenizer.eos_token_id,
                                                                            repetition_penalty=1.06)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(
                generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            # only evalute the generated summary. i.e. the text after task_prompt
            summary_preds = get_summary_after_task_prompt(decoded_preds)
            summary_labels = get_summary_after_task_prompt(decoded_labels)

            summary_preds, summary_labels = postprocess_text(
                summary_preds, summary_labels
            )
            rouge_score.add_batch(predictions=summary_preds,
                                  references=summary_labels)

    # Compute metrics
    result = rouge_score.compute()
    result = {k: round(v, 4) for k, v in result.items()}
    rouge1, rouge2, rougeL, rougeLsum = \
        result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
    # if args.track:
    #     wandb.log({f"eval/rouge1(per n steps)": rouge1,
    #                f"eval/rouge2(per n steps)": rouge2,
    #                f"eval/rougeL(per n steps)": rougeL,
    #                f"eval/rougeLsum(per n steps)": rougeLsum,
    #                })

    return rouge1, rouge2, rougeL, rougeLsum


def get_summary_after_task_prompt(decoded_outputs):
    decoded_summary = []
    for sample_outputs in decoded_outputs:

        summary_start = re.search(task_prompt, sample_outputs).span()[1]
        # find the last period "." after task_prompt
        try:
            summary_end = re.search(
                task_prompt+r'(.*?)(.*)\.', sample_outputs).span()[1]
        except:
            # if there's no period after task_prompt
            summary_end = len(sample_outputs)
        decoded_summary.append(sample_outputs[summary_start:summary_end])

    return decoded_summary


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == "__main__":
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()

    seed = args.seed
    dataset_name_or_path_1 = args.dataset_name_or_path_1
    dataset_name_or_path_2 = args.dataset_name_or_path_2
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    task_prompt = args.task_prompt
    num_virtual_tokens = args.num_virtual_tokens
    prompt_init_text = args.prompt_init_text
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_length = args.max_length
    lr = args.lr
    weight_decay = args.weight_decay
    scheduler_name = args.scheduler_name
    num_warmup_steps = args.num_warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    print_train_every_n_steps = args.print_train_every_n_steps
    eval_steps = args.eval_steps
    track = args.track
    wandb_project_name = args.wandb_project_name
    wandb_entity = args.wandb_entity

    set_seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=prompt_init_text,
        tokenizer_name_or_path=model_name_or_path,
    )

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=lora_rank,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout
    #     )

    peft_model_id = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}"
    run_name = f"{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}_{datetime.now().strftime('%Y%m%d%H%M')}"
    lora_run_name = f"{peft_model_id}_r-{lora_rank}_alpha-{lora_alpha}_{datetime.now().strftime('%Y%m%d%H%M')}"
    wandb_run_name = lora_run_name if peft_config.task_type == "LORA" else run_name

    # Experiment tracking with weights & Bias
    if track:
        run = wandb.init(project=wandb_project_name,
                         entity=wandb_entity,
                         name=wandb_run_name,
                         group=model_name_or_path.split("/")[-1],
                         config=vars(args),
                         sync_tensorboard=True,
                         save_code=True,
                         job_type="train")

    # 1. Data preprocess
    # 1-1. Import the preprocessed data, and then combine them
    dataset_ami_ori = load_dataset(
        "json", data_files=dataset_name_or_path_1, split="train")
    dataset_icsi_ori = load_dataset(
        "json", data_files=dataset_name_or_path_2, split="train")

    meeting_dataset_ori = concatenate_datasets(
        [dataset_ami_ori, dataset_icsi_ori])

    # 1-2. Train / validation / test split based on featrue "split"
    meeting_dataset = dataset_split(meeting_dataset_ori)

    # 1-3. Add task-specific prompt to concatenate text and summary at train and valid split
    processed_train = meeting_dataset["train"].map(add_task_prompt)
    processed_valid = meeting_dataset["valid"].map(add_task_prompt)
    processed_test = meeting_dataset["test"].map(add_task_prompt_test)
    processed_dataset = DatasetDict({
        "train": processed_train,
        "valid": processed_valid,
        "test": processed_test,
    })

    # 1-4. Import model tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1-5. Filter out the example with too long `source`
    # tokenize with on padding and truncation since we want to filter out the data exceed max_length
    tknz_no_padding_ds = processed_dataset.map(
        tokenize_no_maxlen, batched=True)
    filterd_no_padding_ds = tknz_no_padding_ds.filter(
        lambda x: len(x["input_ids"]) <= max_length)

    # After filtered out, tokenize with padding and truncation
    tknz_dataset = filterd_no_padding_ds.map(tokenize, batched=True,
                                             remove_columns=["uid", "id", "text", "summary", "split", "source"])

    # 2. Create DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(tknz_dataset["train"],
                                  collate_fn=data_collator,
                                  batch_size=batch_size,
                                  shuffle=True)
    eval_dataloader = DataLoader(tknz_dataset["valid"],
                                 collate_fn=data_collator,
                                 batch_size=batch_size)
    test_dataloader = DataLoader(tknz_dataset["test"],
                                 collate_fn=data_collator,
                                 batch_size=batch_size)

    # 3. Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        device_map='auto',
    )
    model = get_peft_model(model, peft_config)
    print(f"Is model on cuda? {next(model.parameters()).is_cuda}")
    model.print_trainable_parameters()
    calculate_model_size(model, model_name_or_path)

    # 4. Training hyperparameters
    # optimizer and lr scheduler
    no_decay = ["bias", "layernorm.weight"]
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

    # 5. Training loop
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
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

    rouge_score = evaluate.load("rouge")
    completed_steps = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        model.train()
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.detach().float()

            del batch
            gc.collect()
            torch.cuda.empty_cache()

            if step % print_train_every_n_steps == 0:
                accelerator.print(f"steps: [{epoch+1}]{step}/{len(train_dataloader)} |",
                                  f"updated_steps: {completed_steps} |",
                                  f"lr: {lr_scheduler.get_last_lr()[0]:.7f} |",
                                  f"train/loss: {loss.item():.4f}"
                                  )
            if args.track:
                wandb.log({f"train/loss": loss.item(),
                           f"lr": lr_scheduler.get_last_lr()[0]})
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            # if (step % (eval_steps * gradient_accumulation_steps)) == 0 or (step == (len(train_dataloader)-1)):
            #     rouge1, rouge2, rougeL, rougeLsum = eval_fn(eval_dataloader)
            #     accelerator.print(f"eval_rouge1={rouge1} | eval_rouge2={rouge2} | "
            #                       f"eval_rougeL={rougeL} | eval_rougeLsum={rougeLsum}")
            #     model.train()

        avg_train_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(avg_train_loss)

        rouge1, rouge2, rougeL, rougeLsum = eval_fn(eval_dataloader)
        print(f"epoch {epoch+1} result:\n avg_train_loss={avg_train_loss:.4f} | train_ppl={train_ppl:.4f} | "
              f"eval_rouge1={rouge1:.4f} | eval_rouge2={rouge2:.4f} | "
              f"eval_rougeL={rougeL:.4f} | eval_rougeLsum={rougeLsum:.4f}")

        if args.track:
            wandb.log({f"epoch": epoch+1,
                       f"train/avg_loss": avg_train_loss,
                       f"Train/perplexity": train_ppl,
                       f"eval/rouge1": rouge1,
                       f"eval/rouge2": rouge2,
                       f"eval/rougeL": rougeL,
                       f"eval/rougeLsum": rougeLsum,
                       })

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{output_dir}/{peft_model_id}_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d%H%M')}", save_function=accelerator.save)

    if args.track:
        run.finish()
