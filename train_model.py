import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"


import random
import yaml
from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig 
from algorithms.gtpo_training import UnslothGRPOTrainer as GRPOTrainer
import reward_math
from datasets import load_dataset


PatchFastRL("GRPO", FastLanguageModel)

def load_yaml_config(path="config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        print('\n\n-----------------------')
        print(config)
        print('-----------------------\n\n')
    return config

def load_model_tokenizer(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        fast_inference=True,
        max_lora_rank=config["lora_rank"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_rank"],
        use_gradient_checkpointing="unsloth",
        random_state=config["random_seed"],
    )
    reward_math.count_tokenizer = tokenizer
    return model, tokenizer

def get_dataset(tokenizer, config):
    dataset = load_dataset("gsm8k", "main")["train"]

    print(f'======== ORIGINAL DATASET EXAMPLE ========')
    print(dataset[0])

    system_instruction = (
        "You are a helpful assistant for solving math problems. "
        "Given a question, first think step by step between <reasoning> and </reasoning>. "
        "Then, give the final answer between <answer> and </answer>."
    )

    def format_prompt_completion(sample):
        question = sample["question"]
        answer = sample["answer"]

        reasoning_part, answer_only = answer.rsplit("###", 1)
        reasoning_part = reasoning_part.strip()
        answer_only = answer_only.strip()

        formatted_output = f"<reasoning>\n{reasoning_part}\n</reasoning>\n<answer>{answer_only}</answer>"

        return {
            "prompt": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question},
            ],
            "answer": formatted_output,
        }

    formatted = dataset.map(format_prompt_completion)
    formatted = formatted.remove_columns(["question"])  # Rimuovi campi originali

    def is_valid_prompt(sample):
        full_text = "".join([msg["content"] for msg in sample["prompt"]])
        return len(tokenizer(full_text)["input_ids"]) <= config["max_prompt_length"]

    filtered = formatted.filter(is_valid_prompt)
    print(filtered[0])

    print(f"[INFO] FILTERED DATASET SIZE: {len(filtered)}")
    return filtered.shuffle(seed=42)



def get_output_dir(config):
    name_parts = [
        "GRPO-LLAMA-8gen-NOKL"
        #"Marco-Anomaly(0.7)-Entropy-Aware(0.001)-grad_norm(100)"
        # "GRPO"
        # "Anomaly(1.05)-Entropy-Aware(0.000001)"
        # "Anomaly(0.7)-Entropy-Aware(0.000001)"
        # "NoCumProd-ci=4-NoRationale-SuperAdaptive(EMA0.01)-NotDividedbyN_conflict"
        # "W5-GRPO-"
        # "GRPO-SuperAd-W=5-EMA(0.01)"
        # "Rationale-W5-Super-Adaptive(EMA0.01)_Ci(mu=10^-6, k=8)_No_Central_Confict-NO_STD-NO_PUSH-divided_by_ci-",
        f"adam1-{config['adam_beta1']}",
        f"adam2-{config['adam_beta2']}",
        f"weight_decay-{config['weight_decay']}",
        f"gen{config['num_generations']}",
        f"gr_acc-{config['gradient_accumulation_steps']}",
        f"WarmUp{config['warmup_ratio']}",
        f"len{config['max_seq_length']}",
        f"lr{config['learning_rate']}",
        f"BetaKL{config['beta']}"
    ]
    return "_".join(name_parts)

def train(model, tokenizer, dataset, config):
    output_dir = get_output_dir(config)
    training_args = GRPOConfig(
        learning_rate=float(config["learning_rate"]),
        adam_beta1=config["adam_beta1"],
        adam_beta2=config["adam_beta2"],
        weight_decay=config["weight_decay"],
        beta=config["beta"],
        warmup_ratio=config.get("warmup_ratio", 0.05),
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optimizer"],
        logging_steps=config["logging_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_seq_length"] - config["max_prompt_length"],
        num_train_epochs=config["num_train_epochs"],
        # num_iterations=config["num_iterations"],
        save_steps=config["save_steps"],
        max_grad_norm=config["max_grad_norm"],
        report_to=config.get("report_to", []),
        output_dir=output_dir,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_math.scaled_adaptive_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n\n=========== START TRAINING =========\n\n")
    trainer.train()

    model.save_lora(output_dir)
    return model, tokenizer

def main():
    config = load_yaml_config()
    model, tokenizer = load_model_tokenizer(config)
    dataset = get_dataset(tokenizer, config)
    trained_model, tokenizer = train(model, tokenizer, dataset, config)

if __name__ == "__main__":
    main()
