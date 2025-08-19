"""
2025.3.17
2025.3.19
4.48.2
0.15.2
__UNSLOTH_VERSIONING__
"""
from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import *
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from packaging.version import Version
from torch import Tensor
from torch._dynamo import graph_break  # per il debug facoltativo
from torch._dynamo import disable
from torch.nn import functional as F
from transformers import (DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq)
from trl.trainer.grpo_trainer import (LLM, Any, AutoModelForCausalLM,
                                      AutoModelForSequenceClassification,
                                      AutoTokenizer, Dataset, GenerationConfig,
                                      GRPOConfig, GRPOTrainer, IterableDataset,
                                      Optional, PeftConfig, PreTrainedModel,
                                      PreTrainedTokenizerBase,
                                      RepeatRandomSampler, RewardFunc, Sampler,
                                      SamplingParams, SyncRefModelCallback,
                                      Trainer, TrainerCallback, Union,
                                      apply_chat_template,
                                      broadcast_object_list,
                                      create_reference_model, defaultdict,
                                      gather, gather_object,
                                      generate_model_card,
                                      get_comet_experiment_url,
                                      is_conversational,
                                      is_deepspeed_zero3_enabled,
                                      is_peft_model, is_wandb_available,
                                      maybe_apply_chat_template, nn, os, pad,
                                      patch, prepare_deepspeed, set_seed,
                                      textwrap, torch, transformers,
                                      unwrap_model_for_generation, version,
                                      wandb, warnings)

# ────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & constants
# ────────────────────────────────────────────────────────────────────────────
ENT_THRESHOLD: float = 0.7     # entropy cut‑off
ENT_SCALE: float = 0.1        # advantage ← advantage − ENT_SCALE·entropy
W_RAW: float = 2.0             # weight for tokens in conflict
PAD_ID: int = 128004           # padding token id
EPS: float = 1e-6              # numeric guard for logs
# ---------------------------------------------------------------------------
# 0) Per‑completion statistics – entropy, mean KL
# ---------------------------------------------------------------------------


torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}


@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps

# ────────────────────────────────────────────────────────────────────────────
# 0) Per‑completion statistics – entropy & mean KL
# ────────────────────────────────────────────────────────────────────────────

def compute_completion_stats(
    old_logits: torch.Tensor,   # (L, V)
    new_logits: torch.Tensor,   # (L, V)
    input_ids: torch.Tensor,    # (L,)
    mask: torch.Tensor,         # (L,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (entropy, mean_kl) for *one* completion (grad‑free)."""
    mask_f = mask.float()
    n_tok = mask_f.sum().clamp(min=1.0)

    idx = input_ids.unsqueeze(-1)
    old_logp = old_logits.gather(-1, idx).squeeze(-1) - torch.logsumexp(old_logits, dim=-1)
    new_logp = new_logits.gather(-1, idx).squeeze(-1) - torch.logsumexp(new_logits, dim=-1)

    per_kl = torch.exp(old_logp - new_logp) - (old_logp - new_logp) - 1.0
    probs = torch.softmax(new_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(-1)
    entropy_c = (entropy * mask_f).sum() / n_tok
    wandb.log({"mean_entropy": entropy_c})

    return entropy_c, (per_kl * mask_f).sum() / n_tok

# ────────────────────────────────────────────────────────────────────────────
# helper – build delta masks & conflict stats for a *batch*
# ────────────────────────────────────────────────────────────────────────────


def build_conflict_mask(
    ids: torch.Tensor,          # (G,L)
    mask: torch.Tensor,         # (G,L)
    pos_flags: torch.Tensor,    # (G,) bool  (advantage > 0)
    neg_flags: torch.Tensor,    # (G,) bool  (advantage < 0)
    vocab_size: int,
):

    G, L = ids.shape
    device = ids.device

    active_flags: torch.Tensor = pos_flags | neg_flags               # (G,)
    active_idx:   torch.Tensor = active_flags.nonzero(as_tuple=True)[0]  # (G_act,)
    G_act = active_idx.numel()

    print("\n[DEBUG] G =", G, "L =", L)
    print("[DEBUG] active_flags  :", active_flags.tolist())
    print("[DEBUG] active_idx    :", active_idx.tolist())

    if G_act == 0:
        print("[DEBUG] Nessuna completion attiva. Ritorno valori di default.")
        delta_default  = mask.float()              
        conflict_zero  = torch.zeros_like(mask, dtype=torch.float)
        seq_count_zero = torch.zeros(G, dtype=torch.long, device=device)
        return delta_default, conflict_zero, seq_count_zero

    ids_use   = ids[active_idx]          # (G_act, L)
    mask_use  = mask[active_idx]         # (G_act, L)
    pos_use   = pos_flags[active_idx]    # (G_act,)
    neg_use   = neg_flags[active_idx]    # (G_act,)

    print("[DEBUG] pos_use       :", pos_use.tolist())
    print("[DEBUG] neg_use       :", neg_use.tolist())

    # ------------------------------------------------------------------
    # 1) Conflict FORWARD ----------------------------------------------
    # ------------------------------------------------------------------
    step_idx = torch.arange(L, device=device).unsqueeze(0)           # (1,L)
    keys     = (step_idx * vocab_size + ids_use).view(-1)            # (G_act·L,)

    rpt_pos = pos_use.repeat_interleave(L)   # (G_act·L,)
    rpt_neg = neg_use.repeat_interleave(L)   # (G_act·L,)

    freq_pos = torch.bincount(keys[rpt_pos], minlength=vocab_size * L).clamp_max(1)
    freq_neg = torch.bincount(keys[rpt_neg], minlength=vocab_size * L).clamp_max(1)

    conflict_raw = (freq_pos.bool() & freq_neg.bool())[keys].view(G_act, L)
    print("[DEBUG] conflict_raw (forward) shape:", conflict_raw.shape)

    valid_mask   = (ids_use != PAD_ID)
    initial_run  = torch.cumprod(conflict_raw.to(torch.int), dim=1).bool()
    conflict_fwd = initial_run & valid_mask                           # (G_act,L)

    w_conf_fwd = torch.where(conflict_fwd,
                             torch.tensor(W_RAW, device=device),
                             torch.ones((), device=device))
    delta_fwd  = torch.where(
        ~conflict_fwd,
        1.0,
        torch.where(pos_use.unsqueeze(1), w_conf_fwd, 0.0),
    ) * mask_use

    # ------------------------------------------------------------------
    # 2) Conflict BACKWARD (reverse) ------------------------------------
    # ------------------------------------------------------------------
    seq_len  = mask_use.sum(-1)                # (G_act,)
    pad_len  = L - seq_len

    ids_flip  = ids_use.flip(-1)
    idx_shift = (step_idx + pad_len.unsqueeze(1)) % L
    ids_rev   = ids_flip.gather(1, idx_shift)

    step_rev = torch.arange(L, device=device).flip(0).unsqueeze(0)   # (1,L)
    keys_rev = (step_rev * vocab_size + ids_rev).view(-1)

    freq_pos_r = torch.bincount(keys_rev[rpt_pos], minlength=vocab_size * L).clamp_max(1)
    freq_neg_r = torch.bincount(keys_rev[rpt_neg], minlength=vocab_size * L).clamp_max(1)

    conflict_raw_rev = (freq_pos_r.bool() & freq_neg_r.bool())[keys_rev].view(G_act, L)
    print("[DEBUG] conflict_raw_rev shape:", conflict_raw_rev.shape)

    initial_run_rev = torch.cumprod(conflict_raw_rev.to(torch.int), dim=1).bool()
    conflict_rev    = initial_run_rev & valid_mask

    w_conf_rev = torch.where(conflict_rev,
                             torch.tensor(W_RAW, device=device),
                             torch.ones((), device=device))
    delta_bw_rev = torch.where(
        ~conflict_rev,
        1.0,
        torch.where(pos_use.unsqueeze(1), w_conf_rev, 0.0),
    )

    idx_unshift  = (step_idx - pad_len.unsqueeze(1)) % L
    delta_bw_rot = delta_bw_rev.gather(1, idx_unshift)
    delta_bw_use = delta_bw_rot.flip(-1) * mask_use

    # ------------------------------------------------------------------
    # 3) Final Mask and Stats
    # ------------------------------------------------------------------
    delta_final_use = delta_fwd * delta_bw_use                        # (G_act,L)

    def pretty_mask(mask_tensor: torch.Tensor) -> str:
        symbols = {0: "·", 1: "▒", 2: "█"}
        return "".join(symbols.get(int(v), "?") for v in mask_tensor.tolist())

    for i, g in enumerate(active_idx.tolist()):
        print(f"\n[DEBUG] completion original index {g} (subset row {i})")
        print("FWD :", pretty_mask(delta_fwd[i]))
        print("BWD :", pretty_mask(delta_bw_use[i]))
        print("FIN :", pretty_mask(delta_final_use[i]))

    total_conflict_use = (conflict_fwd | conflict_rev) & mask_use.bool()
    n_conflict_seq_use = total_conflict_use.sum(-1).long().clamp_min(1)  # (G_act,)

   
    delta_final      = mask.float().clone()      # default 1 (inattive)
    total_conflict   = torch.zeros_like(mask, dtype=torch.float)
    n_conflict_seq   = torch.zeros(G, dtype=torch.long, device=device)

    delta_final[active_idx]    = delta_final_use
    total_conflict[active_idx] = total_conflict_use.float()
    n_conflict_seq[active_idx] = n_conflict_seq_use

    return delta_final, total_conflict, n_conflict_seq

# ────────────────────────────────────────────────────────────────────────────
# 1) Loss for a single completion – ratio = 1 with gradient
# ────────────────────────────────────────────────────────────────────────────

def gtpo_compute_loss(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    delta: torch.Tensor,
    beta: float,
    advantages: torch.Tensor,   # scalar tensor already pre‑processed
    n_conflict: torch.Tensor,
):
    mask_f = mask.float(); n_tok = mask_f.sum().clamp(min=1.0)

    idx = input_ids.unsqueeze(-1)
    new_logp = new_logits.gather(-1, idx).squeeze(-1) - torch.logsumexp(new_logits, dim=-1)
    ratio = torch.exp(new_logp - new_logp.detach())  # value 1, carries grad

    adv = advantages.squeeze()
    loss_token = -(ratio * adv * delta)
    loss_disp = (loss_token * mask_f).sum() / n_tok

    with torch.no_grad():
        old_logp = old_logits.gather(-1, idx).squeeze(-1) - torch.logsumexp(old_logits, dim=-1)
        kl = torch.exp(old_logp - new_logp.detach()) - (old_logp - new_logp.detach()) - 1.0
        mean_kl = (kl * mask_f).sum() / n_tok

    loss = (loss_disp / n_conflict.clamp(min=1)).mean()
    return loss, n_tok, mean_kl

# ────────────────────────────────────────────────────────────────────────────
# 2) Autograd wrapper – computes delta/conflict internally
# ────────────────────────────────────────────────────────────────────────────

class UnslothEfficientGTPO(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _new_hidden_states: torch.Tensor,   # (B,G,L*,H)
        _old_hidden_states: torch.Tensor,   # (B,G,L*,H)
        lm_head: torch.Tensor,              # weight matrix (V,H)
        comp_ids: torch.Tensor,             # (B,G,L)
        comp_mask: torch.Tensor,            # (B,G,L)
        advantages: torch.Tensor,           # (G,) 
        rewards: torch.Tensor,              # (G,) 
        beta: float,
        scaler=None,
        n_chunks: int = 1,
    ):
        print("=============== EFFICIENT (CONFLICT INSIDE) ===============")
        device = _new_hidden_states.device
        B, G, Lfull, H = _new_hidden_states.shape
        L = Lfull - 1
        vocab_size = lm_head.size(0)
        lm_w = lm_head.t()  # (H,V)

        # ---- broadcast rewards a (B,G) ----------------------------------
        if rewards.dim() == 1:
            assert rewards.shape[0] == G, f"rewards len {rewards.shape[0]} != G={G}"
            rewards = rewards.unsqueeze(0).expand(B, -1)
        else:
            assert rewards.shape == (B, G), f"rewards shape {tuple(rewards.shape)} != (B,{G})"
        rewards = rewards.to(device)

        if advantages.dim() == 1:
            assert advantages.shape[0] == G, f"advantages len {advantages.shape[0]} != G={G}"
            adv_BG = advantages.unsqueeze(0).expand(B, -1).to(device)  # (B,G)
        elif advantages.dim() == 2:
            assert advantages.shape == (B, G), f"advantages shape {tuple(advantages.shape)} != (B,{G})"
            adv_BG = advantages.to(device)
        else:
            raise ValueError(f"advantages must be 1D or 2D, got {advantages.dim()}D")

        # ---- entropy per completion ------------------------------------
        entropies = torch.zeros((B, G), device=device)
        for b in range(B):
            for g in range(G):
                new_h = _new_hidden_states[b, g]        # (L*,H)
                old_h = _old_hidden_states[b, g]        # (L*,H)
                ids   = comp_ids[b, g]                  # (L,)
                msk   = comp_mask[b, g]                 # (L,)

                e, _ = compute_completion_stats(
                    torch.matmul(old_h, lm_w)[:-1],
                    torch.matmul(new_h, lm_w)[:-1],
                    ids,
                    msk,
                )
                entropies[b, g] = e

        print(f'ENTROPIES : {entropies}')
        low_ent = entropies <= ENT_THRESHOLD  # (B,G)

        # ---- build delta/conflict ---------------------------------------
        delta_masks    = torch.zeros_like(comp_mask, dtype=torch.float32)  # (B,G,L)
        conflict_masks = torch.zeros_like(comp_mask, dtype=torch.float32)  # (B,G,L)
        n_conflict_seq = torch.zeros((B, G), dtype=torch.long, device=device)

        for b in range(B):
            adv_b = adv_BG[b]                # (G,)
            pos_flags_b = (adv_b > 0)        # (G,)
            neg_flags_b = (adv_b < 0)        # (G,)

            print(f'POS FLAGS : {pos_flags_b}')
            print(f'NEG FLAGS : {neg_flags_b}')

            delta_b, conf_b, nconf_b = build_conflict_mask(
                comp_ids[b], comp_mask[b], pos_flags_b, neg_flags_b, vocab_size
            )

            # ─── debug / logging ─────────────────────────────────────────
            active_idx = (adv_b != 0).nonzero(as_tuple=True)[0]
            print(f"Batch {b} – completions con conflitto attivo: {active_idx.tolist()}")
            for g in range(G):
                adv_val = float(adv_b[g].item())
                tag = "+" if adv_val > 0 else ("-" if adv_val < 0 else "0")
                print(f"   completion {g}: adv={tag}  n_conflict={int(nconf_b[g])}")
            # ─────────────────────────────────────────────────────────────

            delta_masks[b]    = delta_b
            conflict_masks[b] = conf_b
            n_conflict_seq[b] = nconf_b

        # ---- entropy shaping sugli advantages ---------------------------
        adv_BG = torch.where(
            low_ent,
            adv_BG - ENT_SCALE * entropies,
            torch.zeros_like(adv_BG),
        )

        # ---- gradient accumulation --------------------------------------
        grad_inputs = torch.empty_like(_new_hidden_states)
        scaling = scaler.get_scale() if scaler is not None else 1.0
        acc_loss = torch.zeros(1, device=device)
        acc_len  = torch.zeros(1, device=device)
        acc_kl   = torch.zeros(1, device=device)

        def compute_loss(new_h, old_h, ids, msk, dlt, adv, rew, c_mask, n_conf, scl):
            new_logits = torch.matmul(new_h, lm_w)[:-1]
            old_logits = torch.matmul(old_h, lm_w)[:-1]
            loss_, ln_, kl_ = gtpo_compute_loss(
                old_logits, new_logits, ids, msk, dlt, beta, adv, n_conf
            )
            return loss_ * scl, (loss_.detach(), ln_, kl_)

        @torch.compile(fullgraph=False)
        def accumulate(new_h, old_h, ids, msk, dlt, adv, rew, c_mask, n_conf, scl):
            (g,), (l, (raw, ln, kl)) = torch.func.grad_and_value(
                compute_loss, argnums=(0,), has_aux=True
            )(new_h, old_h, ids, msk, dlt, adv, rew, c_mask, n_conf, scl)
            acc_loss.add_(raw); acc_len.add_(ln); acc_kl.add_(kl)
            return g

        for b in range(B):
            for g in range(G):
                grad_inputs[b, g] = accumulate(
                    _new_hidden_states[b, g], _old_hidden_states[b, g],
                    comp_ids[b, g], comp_mask[b, g], delta_masks[b, g],
                    adv_BG[b, g].unsqueeze(0),
                    rewards[b, g].unsqueeze(0),
                    conflict_masks[b, g],
                    n_conflict_seq[b, g].unsqueeze(0),
                    scaling,
                )

        ctx.save_for_backward(grad_inputs)

        for b in range(B):
            for g in range(G):
                grad_norm = grad_inputs[b, g].norm().item()
                print(f"[b={b}] completion {g}: grad_norm = {grad_norm:.6f}")

        return acc_loss, acc_len, acc_kl

    @staticmethod
    def backward(ctx, grad_out, dlen, dkl):
        (grad_input,) = ctx.saved_tensors
        return (grad_input,) + (None,) * 12


def GTPO_accumulated_loss(
    trainer,
    input_ids: torch.Tensor,
    logits_to_keep: int,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,  
    rewards: torch.Tensor,
    *,                         
    n_chunks: int = -1,
):

    device = input_ids.device
    bsz, _ = input_ids.shape

    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1:
        n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors) - 1)]

    mixed_dtype = (
        torch.float16
        if os.getenv("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
        else torch.bfloat16
    )
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    if rewards.dim() == 1:                                      # case B = 1
        rewards = rewards.unsqueeze(0)                          # (1,G)
        completion_mask = completion_mask.unsqueeze(0)
        completion_input_ids = completion_input_ids.unsqueeze(0)

    B, G = rewards.shape
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type="cuda", dtype=mixed_dtype):
        with torch.no_grad(), trainer.accelerator.unwrap_model(
            trainer.model, keep_fp32_wrapper=False
        ).disable_adapter():
            old_hidden_states = trainer.model(
                input_ids=input_ids, logits_to_keep=logits_to_keep + 1
            ).logits
        new_hidden_states = trainer.model(
            input_ids=input_ids, logits_to_keep=logits_to_keep + 1
        ).logits

    Lfull, H = new_hidden_states.shape[1:]
    L = Lfull - 1
    new_hidden_states = new_hidden_states.view(B, G, Lfull, H).contiguous()
    old_hidden_states = old_hidden_states.view(B, G, Lfull, H).contiguous()

    comp_ids  = completion_input_ids.view(B, G, L)
    comp_mask = completion_mask.view(B, G, L)

    loss, completion_length, mean_kl = UnslothEfficientGTPO.apply(
        new_hidden_states, old_hidden_states, lm_head,
        comp_ids, comp_mask,          
        advantages, rewards, trainer.beta, trainer.accelerator.scaler, n_chunks,
    )

    return loss, completion_length, mean_kl


# @torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)
def GTPO_compute_loss_debug(old_logits, new_logits, input_ids, mask, beta, advantages):
    print(f'========================DEBUG FUNCTION=============================', flush=True)
    # All Unsloth Zoo code licensed under LGPLv3
    old_logits = old_logits.to(torch.float32)
    new_logits = new_logits.to(torch.float32)
    input_ids  = input_ids.unsqueeze(-1)

    # x_i - logsumexp(x_i)
    old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    old = old_x - torch.logsumexp(old_logits, dim = -1)
    new = new_x - torch.logsumexp(new_logits, dim = -1)

    # Reverse KL
    kl_i = torch.exp(old - new) - (old - new) - 1.0

    loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    print(f'LOSS for each TOKEN before KL:', flush=True)
    print(loss_i, flush=True)
    loss_i = -(loss_i - beta * kl_i)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # See https://github.com/huggingface/trl/pull/2881
    loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    loss = loss_per_reward.mean()
    
    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()
    pass
    return loss, completion_length, mean_kl

def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params

@dataclass
class UnslothGTPOConfig(GRPOConfig):
    """
    
    Configuration class for the [`GTPOTrainer`].

    Only the parameters specific to GTPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GTPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log the completions during training.
    
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 0.1,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        use_ipex = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = False,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = '',
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = None,
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        evaluation_strategy = None,
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = False,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        dispatch_batches = None,
        split_batches = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        eval_use_gather_object = False,
        average_tokens_across_devices = False,
        model_init_kwargs = None,
        max_prompt_length = 512,
        num_generations = 8,
        temperature = 0.9,
        max_completion_length = 256,
        ds3_gather_for_generation = True,
        use_vllm = False,
        vllm_device = 'auto',
        vllm_gpu_memory_utilization = 0.9,
        vllm_dtype = 'auto',
        vllm_max_model_len = None,
        beta = 0.04,
        reward_weights = None,
        sync_ref_model = False,
        ref_model_mixup_alpha = 0.9,
        ref_model_sync_steps = 64,
        log_completions = False,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
        if learning_rate < 1e-7: raise FloatingPointError(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: raise OverflowError(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        div = per_device_train_batch_size // num_generations
        if div * num_generations != per_device_train_batch_size:
            print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\nWe will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))
            per_device_train_batch_size = num_generations
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            use_ipex = use_ipex,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            evaluation_strategy = evaluation_strategy,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            dispatch_batches = dispatch_batches,
            split_batches = split_batches,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            model_init_kwargs = model_init_kwargs,
            max_prompt_length = max_prompt_length,
            num_generations = num_generations,
            temperature = temperature,
            max_completion_length = max_completion_length,
            ds3_gather_for_generation = ds3_gather_for_generation,
            use_vllm = use_vllm,
            vllm_device = vllm_device,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_dtype = vllm_dtype,
            vllm_max_model_len = vllm_max_model_len,
            beta = beta,
            reward_weights = reward_weights,
            sync_ref_model = sync_ref_model,
            ref_model_mixup_alpha = ref_model_mixup_alpha,
            ref_model_sync_steps = ref_model_sync_steps,
            log_completions = log_completions,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

class _UnslothGTPOTrainer(Trainer):
    """"""

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm') and (getattr(args, 'use_vllm', False) == False): args.use_vllm = True
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GTPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if False:
            model = model

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GTPO paper
        self.num_generations = args.num_generations  # = G in the GTPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        print(f'GLOBAL BATCH SIZE: {global_batch_size}, NUM PROCESSES: {num_processes}')
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            self.llm = model.vllm_engine; self._last_loaded_step = 0; self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,**getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {}),)
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        if os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return None # Unsloth efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, '_autocast_dtype'):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': self._autocast_dtype = torch.float16
        with torch.amp.autocast(device_type = 'cuda', dtype = self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            return logits
            # return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
        pass

    def _move_model_to_vllm(self, *args, **kwargs): return None

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False, lora_request = self.model.load_lora('grpo_trainer_lora_model', load_tensors = True))
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode(), torch.amp.autocast(device_type = 'cuda', dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) if not torch.is_autocast_enabled('cuda') else nullcontext())if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode(), torch.amp.autocast(device_type = 'cuda', dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) if not torch.is_autocast_enabled('cuda') else nullcontext())if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        #rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        #std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) #/ (std_grouped_rewards + 1e-4)
        print(f'REWARDS: {rewards}')
        print(f'MEAN : {mean_grouped_rewards}')
        print(f'ORIGINAL ADVANTAGES : {advantages}')

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())

        print(f'COMPLETION MASK SUM : {completion_mask.sum(dim=1)}')
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards
        }

    #     return loss
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep
        
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)


        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]
        rewards = inputs["rewards"]
        input_ids = input_ids[:, -logits_to_keep:]
        if per_token_logps is not None:
            loss, completion_length, mean_kl = GTPO_compute_loss_debug(
                ref_per_token_logps, per_token_logps, input_ids, completion_mask, self.beta, advantages,
            )
        else:
            loss, completion_length, mean_kl = GTPO_accumulated_loss(
                self, _input_ids, logits_to_keep, completion_mask, advantages, rewards,
                n_chunks = self.args.unsloth_num_chunks,
            )


        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
class UnslothGTPOTrainer(_UnslothGTPOTrainer):
    """
    
    Trainer for the Group Relative Policy Optimization (GTPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GTPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GTPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GTPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    
    """
    def __init__(
        self,
        model,
        reward_funcs,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_classes = None,
        callbacks = None,
        peft_config = None,
        **kwargs
    ):
        if args is None: args = UnslothGRPOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        use_fp16 = getattr(args, 'fp16', False)
        force_float32 = False
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training()
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        other_metrics = []
        if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]
        else: _reward_funcs = reward_funcs
        for reward_func in _reward_funcs:
            try:
                reward_func_name = reward_func.__name__
                other_metrics.append(f'rewards/{reward_func_name}')
            except: pass
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('grpo_trainer', other_metrics)
        
        super().__init__(
            model = model,
            reward_funcs = reward_funcs,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_classes = reward_processing_classes,
            callbacks = callbacks,
            peft_config = peft_config,**kwargs)
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        
pass
