import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_utils import generate_tasks, generate_dataset, generate_context_prompt


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(2)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def load_model_and_tokenizer(base_model: str = "Llama-3-8B", cluster: int = 3, load_bit: int = 8):
    """Load model and tokenizer based on the provided configuration."""
    if base_model == "Llama-3-8B":
        path = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_8bit = load_bit == 8
    load_in_4bit = load_bit == 4
    model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    if load_bit not in [8, 4]:
        model.to(device)
    print(f"Loaded model and tokenizer from {path}")
    return model, tokenizer


def plot_decision_boundary(
    X_train, y_train, xx1, xx2, predictions, model_name="llama3-8b", num_in_context=50, grid_size=50
):
    """Plot the decision boundary for the given data and predictions."""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.contourf(xx1, xx2, predictions, alpha=0.8, cmap=ListedColormap(["#FF9999", "#9999FF"]))
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=36,
        edgecolor="navy",
        cmap=ListedColormap(["#FF0000", "#0000FF"]),
    )

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(f"{model_name}\n{num_in_context} In-Context Examples", fontsize=32)
    ax.set_xlabel("Feature 1", fontsize=26)
    ax.set_ylabel("Feature 2", fontsize=26)

    plt.savefig(
        f"{model_name}_{num_in_context}incontext.png",
        bbox_inches="tight",
    )
    print(f"Decision boundary plot saved as {model_name}_{num_in_context}incontext.png")


def create_prompts(args, system_prompt, context_prompt, query_prompt, inputs):
    if "instruct" in args.model_name:
        # Llama instruction prompt format
        prompts = [
            f"### Instructions:\n"
            f"{system_prompt}\n"
            f"### Input:\n"
            f"{context_prompt}\n"
            f"{query_prompt}\n"
            f"Input: {inp}\n"
            f"### Response:\n"
            f"Label: "
            for inp in inputs
        ]

    else:
        prompts = [
            f"{system_prompt}\n{context_prompt}\n{query_prompt}\nInput: {inp}\nLabel: " for inp in inputs
        ]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate a binary classification dataset and split it.")
    parser.add_argument("--model_name", type=str, default="Llama-3-8B", help="Model name")
    parser.add_argument(
        "--num_in_context", type=int, default=128, help="Number of samples per class for in-context learning"
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Grid size for decision boundary plotting")
    parser.add_argument("--num_test_samples", type=int, default=100, help="Number of test examples")
    parser.add_argument("--load_bit", type=int, default=8, help="Bit configuration for loading the model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="01", help="Experiment name")
    parser.add_argument("--num_dimensions", type=int, default=2, help="Number of dimensions for the samples")
    parser.add_argument("--num_tasks", type=int, default=1000, help="Number of tasks to generate")
    parser.add_argument("--num_samples_per_task", type=int, default=800, help="Number of samples per task")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train-test split ratio")
    parser.add_argument(
        "--data_type", type=str, default="linear", help="Type of data to generate, linear, circle or moon"
    )
    parser.add_argument(
        "--class_sep", type=float, default=1.0, help="Class separation for linearly separable data"
    )
    parser.add_argument(
        "--circle_factor", type=float, default=0.5, help="Circle factor for circular data generation"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    class_names_dict = {
        "foobar": ["Foo", "Bar"],
        "01": ["0", "1"],
        "AandB": ["A", "B"],
        "reverse_foobar": ["Bar", "Foo"],
        "PosNeg": ["Positive", "Negative"],
        "yesno": ["Yes", "No"],
    }

    if args.exp_name not in class_names_dict:
        raise ValueError(f"Unknown experiment name: {args.exp_name}")

    class_names = class_names_dict[args.exp_name]

    # Generate tasks and split into training and testing sets
    dataset_x, dataset_y = generate_tasks(
        num_tasks=args.num_tasks,
        num_samples_per_task=args.num_samples_per_task,
        num_dimensions=args.num_dimensions,
        seed=args.seed,
        data_type=args.data_type,
        class_sep=args.class_sep,
        factor=args.circle_factor,
    )
    meta_train_X, meta_test_X, meta_train_y, meta_test_y = train_test_split(
        dataset_x, dataset_y, train_size=args.train_ratio
    )

    print(f"Meta_train_X shape: {meta_train_X.shape}")
    print(f"Meta_test_X shape: {meta_test_X.shape}")
    print("-" * 50)

    system_prompt = f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data. Answer with only one of the labels '{class_names[0]}' and '{class_names[1]}'."
    query_prompt = "What is the label for this input?"

    context_x, context_y, query_x, query_y = generate_dataset(args, meta_train_X, meta_train_y)

    print("-" * 10, "Context X, Y shapes:", context_x.shape, context_y.shape, query_x.shape, query_y.shape)

    model, tokenizer = load_model_and_tokenizer(base_model=args.model_name, load_bit=args.load_bit)
    desired_task_idx = [0]  # only on the first task for demo

    for task_idx in desired_task_idx:

        task_context_x = context_x[task_idx]
        task_context_y = context_y[task_idx]

        x1_min, x1_max = task_context_x[:, 0].min() - 1, task_context_x[:, 0].max() + 1
        x2_min, x2_max = task_context_x[:, 1].min() - 1, task_context_x[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, args.grid_size), np.linspace(x2_min, x2_max, args.grid_size)
        )
        xx1_flat, xx2_flat = xx1.ravel(), xx2.ravel()
        inputs = [f"{int(x)} {int(y)}" for x, y in zip(xx1_flat, xx2_flat)]

        context_prompt = generate_context_prompt(X=task_context_x, y=task_context_y, class_names=class_names)
        prompts = create_prompts(args, system_prompt, context_prompt, query_prompt, inputs)

        # Store the KV cache for the in-context examples to speed up.
        inputs_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False)["input_ids"]
        max_input_len = inputs_ids.shape[1] + 10  # 10 is a random Buffer for tokenization differences

        predictions = np.zeros(xx1_flat.shape[0])
        logits_pred = np.zeros((xx1_flat.shape[0], 2))

        prompt_input_ids = tokenizer(prompts[0], return_tensors="pt", padding=True, truncation=False)[
            "input_ids"
        ]
        in_context_ids = prompt_input_ids[:, :-max_input_len].to(model.device)

        with torch.inference_mode():
            in_context_kv_cache = model(input_ids=in_context_ids, return_dict=True).past_key_values

        # Tokenization robustness. This handles cases where the tokenizer may split "Foo" differently when preceded by a space
        tokens = class_names
        token_ids_foo = tokenizer(tokens[0], return_tensors="pt")["input_ids"][0][-1]
        token_ids_bar = tokenizer(tokens[1], return_tensors="pt")["input_ids"][0][-1]
        token_ids_foo1 = tokenizer(f" {tokens[0]}", return_tensors="pt")["input_ids"][0][-1]
        token_ids_bar1 = tokenizer(f" {tokens[1]}", return_tensors="pt")["input_ids"][0][-1]

        # Probe decision boundary
        for i in tqdm(range(0, len(prompts))):
            batch_prompts = prompts[i : i + 1]

            total_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)[
                "input_ids"
            ].to(model.device)
            this_in_context_ids = total_prompt[:, : len(in_context_ids[0])]
            question_ids = total_prompt[:, len(in_context_ids[0]) :]

            assert torch.equal(this_in_context_ids, in_context_ids)
            with torch.inference_mode():
                in_context_q_kv_cache = model(
                    question_ids[:, :-1], past_key_values=in_context_kv_cache, return_dict=True
                ).past_key_values

                generations = model.generate(
                    input_ids=total_prompt,
                    do_sample=False,
                    max_new_tokens=1,
                    past_key_values=in_context_q_kv_cache,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                logits = generations["scores"][-1]

            logit_bar = max(logits[0, token_ids_bar].item(), logits[0, token_ids_bar1].item())
            logit_foo = max(logits[0, token_ids_foo].item(), logits[0, token_ids_foo1].item())
            generated_texts = tokenizer.batch_decode(
                generations["sequences"][:, -1:], skip_special_tokens=True
            )
            for idx, (generated_text, x_val, y_val) in enumerate(
                zip(generated_texts, xx1_flat[i : i + 1], xx2_flat[i : i + 1])
            ):
                # Check if the generated text contains the class names, if not, use the logit to predict
                if class_names[0].lower() in generated_text.lower():
                    predictions[i + idx] = 0
                elif class_names[1].lower() in generated_text.lower():
                    predictions[i + idx] = 1
                else:
                    if logit_bar > logit_foo:
                        predictions[i + idx] = 1
                        logits_pred[i + idx] = [logit_bar, logit_foo]
                    else:
                        predictions[i + idx] = 0
                        logits_pred[i + idx] = [logit_foo, logit_bar]

        llm_predictions = predictions.reshape(xx1.shape)
        plot_decision_boundary(
            task_context_x,
            task_context_y,
            xx1,
            xx2,
            llm_predictions,
            model_name=args.model_name,
            num_in_context=args.num_in_context,
            grid_size=args.grid_size,
        )


if __name__ == "__main__":
    main()
