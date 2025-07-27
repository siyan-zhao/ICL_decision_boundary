from transformers.modeling_outputs import BaseModelOutputWithPast

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import random
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
)


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


@dataclass
class FewshotFinetuneCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_text = [example for example in examples]
        tokenized_full_text = self.tokenizer(full_text, padding=True, truncation=True, return_tensors="pt")
        tokenized_full_text["labels"] = tokenized_full_text["input_ids"].clone()
        return tokenized_full_text


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_model_and_tokenizer(
    base_model: str = "llama3-8b",
    cluster: int = 3,
    loadbit: int = 8,
    use_lora: bool = False,
    resume_path: Optional[str] = None,
):
    # Load tokenizer and model
    if base_model == "llama2":
        path = "/localhome/data/ckpts/shared/llama-2-7b-chat-hf"  
    elif base_model == "llama2-13b":
        path = "/localhome/data/ckpts/shared/llama-2-13b-chat-hf"
    elif base_model == "mistral":
        path = "/localhome/data/ckpts/shared/Mistral-7B-v0.1" 
    elif base_model == "llama1b":
        path = "princeton-nlp/Sheared-LLaMA-1.3B"
    elif base_model == "llama3-8b":
        path = "/localhome/data/ckpts/shared/Meta-Llama-3-8B"
    elif base_model == "llama3-8b_instruct":
        path = "/localhome/data/ckpts/shared/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_8bit = loadbit == 8
    load_in_4bit = loadbit == 4
    model = AutoModelForCausalLM.from_pretrained(
        path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit  # , device_map="auto"
    )
    if use_lora:
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=6,
            lora_alpha=16,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        if resume_path is not None:
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
            print("-" * 20, "\nResuming from", resume_path, "\n", "-" * 20)
        else:
            model = get_peft_model(model, lora_config)

    if loadbit != 8 and loadbit != 4:
        model.to(device)

    return model, tokenizer


def generate_x_y(num_samples, num_dimensions, seed):
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_dimensions,
        n_informative=num_dimensions // 2,
        n_redundant=0,  # no redundant features
        n_clusters_per_class=1,  # each class is a single cluster
        weights=[0.5, 0.5],  # equal distribution of classes
        flip_y=0,  # no noise
        shuffle=True,
        random_state=seed,
    )

    # Normalize X to [0, 1] and then scale to [0, 100]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 100 * (X - X_min) / (X_max - X_min)
    return X, y


def generate_tasks(num_tasks, num_samples_per_task, num_dimensions, seed):
    # Create empty arrays to store X and Y data
    X_data = np.zeros((num_tasks, num_samples_per_task, num_dimensions))
    Y_data = np.zeros((num_tasks, num_samples_per_task))

    for i in range(num_tasks):
        X, y = generate_x_y(num_samples=num_samples_per_task, num_dimensions=num_dimensions, seed=seed + i)
        X_data[i] = X
        Y_data[i] = y

    return X_data, Y_data


def generate_context_prompt(X, y, class_names):
    y_named = [class_names[int(label)] for label in y]

    prompt = ""
    for features, label in zip(X, y_named):
        features_str = " ".join(f"{int(num)}" for num in np.round(features))
        prompt += f"Input: {features_str}\nLabel: {label}\n"
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate a binary classification dataset and split it.")
    # dataset arguments
    parser.add_argument("--ict_num", type=int, default=64, help="Number of samples per class")
    parser.add_argument("--loadbit", type=int, default=16, help="Bit loading configuration")
    parser.add_argument("--data_dim", type=int, default=2, help="Number of dimensions of data points")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="foobar", help="Experiment name")
    parser.add_argument("--num_tasks", type=int, default=1000, help="Number of tasks")
    parser.add_argument("--num_samples", type=int, default=1024, help="Number of samples per task")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    parser.add_argument("--dim", type=int, default=2, help="Number of dimensions")
    parser.add_argument("--model_name", type=str, default="llama3-8b", help="Model name")
    # training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--eval_step", type=int, default=50, help="Evaluation step")
    parser.add_argument("--train_bf16", action="store_true", help="Train in bf16")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")

    args = parser.parse_args()
    model_save_path = "/home/siyanz/llm_inductivebias/ckpts"
    set_seed(args.seed)
    if args.exp_name == "foobar":
        class_names = ["Foo", "Bar"]
    elif args.exp_name == "01":
        class_names = ["0", "1"]
    elif args.exp_name == "AandB":
        class_names = ["A", "B"]
    elif args.exp_name == "reverse_foobar":
        class_names = ["Bar", "Foo"]

    dataset_x, dataset_y = generate_tasks(
        num_tasks=args.num_tasks,
        num_samples_per_task=args.num_samples,
        num_dimensions=args.dim,
        seed=args.seed,
    )
    print(dataset_x.shape, dataset_y.shape)
    meta_train_X, meta_test_X, meta_train_y, meta_test_y = train_test_split(
        dataset_x, dataset_y, train_size=args.train_size
    )
    print(f"Meta_train_X shape: {meta_train_X.shape}", f"Meta_test_X shape: {meta_test_X.shape}")
    system_prompt = f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data. Answer with only one of the labels '{class_names[0]}' and '{class_names[1]}'."
    query_prompt = "What is the label for this input?"
    all_prompts = []
    for task_idx, (task_x, task_y) in enumerate(zip(meta_train_X, meta_train_y)):
        num_per_class = args.ict_num // 2
        class_0_indices = np.where(task_y == 0)[0][:num_per_class]
        class_1_indices = np.where(task_y == 1)[0][:num_per_class]
        context_indices = np.concatenate([class_0_indices, class_1_indices])
        np.random.shuffle(context_indices)

        context_x = task_x[context_indices]
        context_y = task_y[context_indices]

        query_indices = np.setdiff1d(
            np.arange(len(task_y)), context_indices
        )  # Efficient way to get non-overlapping indices

        queries_x = task_x[query_indices]
        queries_y = task_y[query_indices]

        # assert no overlap between context and query
        assert len(set(context_indices) & set(query_indices)) == 0

        context_prompt = generate_context_prompt(X=context_x, y=context_y, class_names=class_names)

        if "instruct" in args.model_name:
            # Llama instruction prompt format
            prompts = [
                f"### Instructions:\n"
                f"{system_prompt}\n"
                f"### Input:\n"
                f"{context_prompt}\n"
                f"{query_prompt}\n"
                f"Input: {int(q_x[0])} {int(q_x[1])}\n"
                f"### Response:\n"
                f"Label: {class_names[int(q_y)]}"
                for q_x, q_y in zip(queries_x, queries_y)
            ]

        else:
            prompts = [
                f"{system_prompt}\n{context_prompt}\n{query_prompt}\nInput: {int(q_x[0])} {int(q_x[1])}\nLabel: {class_names[int(q_y)]}"
                for q_x, q_y in zip(queries_x, queries_y)
            ]
        all_prompts.extend(prompts)
    print("using lora:", args.use_lora)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.model_name, loadbit=args.loadbit, use_lora=args.use_lora
    )
    wandb.init(entity="icl", project="llm_inductive_bias", name=args.exp_name, config=args)
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accum_steps,  # Further increase gradient accumulation
        output_dir=f"{model_save_path}/{args.exp_name}",
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        save_strategy="steps",
        save_steps=args.eval_step,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        # stablize training:
        # max_grad_norm=1,
        # weight_decay=0.001,
        # warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        bf16=args.train_bf16,
        metric_for_best_model="eval_accuracy",
    )
    train_prompts = all_prompts[: int(args.train_size * len(all_prompts))]
    eval_prompts = all_prompts[int(args.train_size * len(all_prompts)) :]

    train_dataset = PromptDataset(train_prompts)
    eval_dataset = PromptDataset(eval_prompts)
    print(len(train_dataset), len(eval_dataset), "train and eval dataset lengths")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=FewshotFinetuneCollator(tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
