import re

import numpy as np
import wandb

step_counter = 0  

def extract_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>$", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_xml_reasoning(text: str) -> str:
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def compute_format_reward(text):
    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>$", text, re.DOTALL))

    if has_reasoning and has_answer:
        return 10.0

    total, correct_a, correct_r = 0, 0, 0
    if has_reasoning:
        partial_tags = any(re.search(tag, text) for tag in [r"<answer>", r"</answer>"])
        correct_r = 1.5 if partial_tags else 1.0
    
    if has_answer:
        partial_tags = any(re.search(tag, text) for tag in [r"<reasoning>", r"</reasoning>"])
        correct_a = 1.5 if partial_tags else 1.0

    total = correct_r + correct_a
    return total

def compute_correctness_reward(pred, true):
    
    pred_answer = extract_answer(pred)
    true_answer = extract_answer(true)

    if pred_answer is not None and true_answer is not None:
        try:
            pred_val = float(pred_answer)
            true_val = float(true_answer)
            if np.isclose(pred_val, true_val, atol=1e-2):
                return 10.0
            elif str(true_answer) in str(pred_answer):
                return 10.0
        except Exception as e:
            print(f' ERROR {e}')
            if str(true_answer) in str(pred_answer):
                return 10.0
            
    return 0.0

def final_reward(prompts, completions, answer):
    global step_counter 

    step_counter += 1
    rewards = []
    format_rewards = []
    correctness_rewards = []
    comp_gen = 0
    for completion, reference in zip(completions, answer):
        print(f'+++++++++++++++++ STEP: {step_counter} +++++++++++++++++')
        comp_gen += 1
        completion = completion[0]['content']
        print(f' ============   COMP - {comp_gen} =============')
        print(f'COMPLETION: {completion}')
        # print(f'ANSWER: {reference}')
        format_reward = compute_format_reward(completion)
        if format_reward < 10:
            correctness_reward = 0
        else:
            correctness_reward = compute_correctness_reward(completion, reference)
        total_reward = format_reward + correctness_reward
        print(f'    - Format Reward : {format_reward}')
        print(f'    - Correctness R : {correctness_reward}')
        print(f'    - Total Reward  : {total_reward}')

        rewards.append(total_reward)
        format_rewards.append(format_reward)
        correctness_rewards.append(correctness_reward)

    # Loggare su wandb
    wandb.log({
        "reward/formatting_avg": np.mean(format_rewards),
        "reward/correctness_avg": np.mean(correctness_rewards),
        "reward/total_avg": np.mean(rewards),
    })

    return rewards
