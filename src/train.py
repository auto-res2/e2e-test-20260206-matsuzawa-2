import os
from pathlib import Path

def setup_cache_env() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("DATASETS_CACHE", str(cache_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return str(cache_dir)


setup_cache_env()

import math
import random
import time
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler
from transformers import get_linear_schedule_with_warmup

from src.model import MathSolverModel, MethodRunner
from src.preprocess import numbers_match, parse_number_from_text, prepare_dataset


def ensure_run_node(cfg: DictConfig) -> None:
    if not hasattr(cfg, "run_id"):
        raise ValueError("run_id is missing from the configuration.")
    if not hasattr(cfg, "run") or isinstance(cfg.run, str):
        cfg.run = OmegaConf.create({"run_id": cfg.run_id})
    elif not hasattr(cfg.run, "run_id"):
        cfg.run.run_id = cfg.run_id


def apply_mode(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "optuna"):
            cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.max_batches = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    return cfg


def sync_timeout_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "method_params"):
        return cfg
    if not hasattr(cfg.method_params, "sandbox") or cfg.method_params.sandbox is None:
        cfg.method_params.sandbox = OmegaConf.create({})
    if hasattr(cfg.method_params, "exec_timeout_s") and cfg.method_params.exec_timeout_s is not None:
        cfg.method_params.sandbox.timeout_s = float(cfg.method_params.exec_timeout_s)
    else:
        timeout = getattr(cfg.method_params.sandbox, "timeout_s", 1.0)
        cfg.method_params.exec_timeout_s = float(timeout)
        cfg.method_params.sandbox.timeout_s = float(timeout)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_supervised_loss(
    model_wrapper: MathSolverModel,
    question: str,
    answer_text: str,
    max_label_tokens: int,
) -> torch.Tensor:
    if answer_text is None or str(answer_text).strip() == "":
        raise ValueError("answer_text must be provided for supervised loss computation.")
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device

    prompt = f"Question: {question}\nAnswer:"
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    answer_ids = tokenizer(
        str(answer_text),
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    )["input_ids"].to(device)

    if answer_ids.numel() == 0:
        raise ValueError("Label tokenization produced empty sequence.")

    if max_label_tokens and answer_ids.shape[1] > max_label_tokens:
        answer_ids = answer_ids[:, :max_label_tokens]

    input_ids = prompt_inputs["input_ids"]
    max_pos = getattr(model_wrapper.model.config, "max_position_embeddings", None)
    if max_pos is not None:
        max_allowed = max_pos - input_ids.shape[1] - 1
        max_allowed = max(1, max_allowed)
        if answer_ids.shape[1] > max_allowed:
            answer_ids = answer_ids[:, :max_allowed]

    losses = []
    attention_mask = torch.ones_like(input_ids)
    for idx in range(answer_ids.shape[1]):
        outputs = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        target = answer_ids[:, idx]
        loss = F.cross_entropy(logits, target)
        losses.append(loss)
        next_token = torch.argmax(logits, dim=-1, keepdim=True).detach()
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.ones_like(input_ids)

    return torch.stack(losses).mean()


def create_optimizer(cfg: DictConfig, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
    name = str(cfg.training.optimizer).lower()
    lr = float(cfg.training.learning_rate)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(getattr(cfg.training, "momentum", 0.0))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
):
    warmup_steps = int(getattr(cfg.training, "warmup_steps", 0))
    if warmup_steps <= 0:
        return None
    warmup_steps = min(warmup_steps, total_steps)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def apply_params_to_cfg(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "method_params") or cfg.method_params is None:
        cfg.method_params = OmegaConf.create({})
    for name, value in params.items():
        if name.startswith("max_new_tokens_"):
            if not hasattr(cfg, "model") or cfg.model is None:
                cfg.model = OmegaConf.create({})
            cfg.model[name] = int(value)
            continue
        if hasattr(cfg.model, name):
            cfg.model[name] = value
        elif hasattr(cfg.method_params, name):
            cfg.method_params[name] = value
        else:
            cfg.method_params[name] = value

    if "exec_timeout_s" in params:
        cfg.method_params.exec_timeout_s = float(params["exec_timeout_s"])
        if not hasattr(cfg.method_params, "sandbox") or cfg.method_params.sandbox is None:
            cfg.method_params.sandbox = OmegaConf.create({})
        cfg.method_params.sandbox.timeout_s = float(params["exec_timeout_s"])
    if "max_new_tokens_cot" in params:
        cfg.model.max_new_tokens_cot = int(params["max_new_tokens_cot"])
    if "max_new_tokens_code" in params:
        cfg.model.max_new_tokens_code = int(params["max_new_tokens_code"])
    if "groundedness_mode" in params:
        cfg.method_params.groundedness_mode = params["groundedness_mode"]
    if "float_tol" in params:
        cfg.method_params.float_tol = float(params["float_tol"])
    if "use_repair_round" in params:
        value = params["use_repair_round"]
        if isinstance(value, str):
            value = int(value) if value.isdigit() else value
        cfg.method_params.use_repair_round = bool(value)
    return sync_timeout_cfg(cfg)


def suggest_params(trial: optuna.Trial, search_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for space in search_spaces:
        name = space["param_name"]
        dist = space["distribution_type"]
        if dist == "uniform":
            params[name] = trial.suggest_float(name, space["low"], space["high"])
        elif dist == "loguniform":
            params[name] = trial.suggest_float(name, space["low"], space["high"], log=True)
        elif dist == "int":
            params[name] = trial.suggest_int(name, int(space["low"]), int(space["high"]))
        elif dist == "categorical":
            params[name] = trial.suggest_categorical(name, space["choices"])
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    return params


def get_effective_examples(
    dataset_len: int,
    max_examples: int | None,
    max_batches: int,
    batch_size: int,
) -> int:
    total = dataset_len
    if max_examples is not None:
        total = min(total, max_examples)
    if max_batches > 0:
        total = min(total, max_batches * batch_size)
    return max(0, total)


def run_optuna(cfg: DictConfig, dataset: List[Dict[str, Any]], model_wrapper: MathSolverModel) -> DictConfig:
    if not hasattr(cfg, "optuna") or cfg.optuna.n_trials <= 0:
        return cfg

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=cfg.training.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    eval_samples = int(getattr(cfg.optuna, "eval_samples", 50))
    subset = dataset[: min(eval_samples, len(dataset))]

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        trial_cfg.wandb.mode = "disabled"
        trial_cfg.training.max_batches = -1
        trial_cfg.training.update_model = False
        params = suggest_params(trial, trial_cfg.optuna.search_spaces)
        trial_cfg = apply_params_to_cfg(trial_cfg, params)
        runner = MethodRunner(model_wrapper, trial_cfg)
        metrics, _, _ = evaluate_dataset(
            trial_cfg,
            subset,
            model_wrapper,
            runner,
            wandb_run=None,
            max_examples=len(subset),
            step_offset=0,
            update_model=False,
            optimizer=None,
            scheduler=None,
            log_steps=False,
        )
        return metrics["accuracy"]

    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    cfg = apply_params_to_cfg(cfg, study.best_params)
    return cfg


def evaluate_dataset(
    cfg: DictConfig,
    dataset: List[Dict[str, Any]],
    model_wrapper: MathSolverModel,
    runner: MethodRunner,
    wandb_run: Any,
    max_examples: int | None,
    step_offset: int,
    update_model: bool,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    log_steps: bool,
) -> Tuple[Dict[str, float], Dict[str, int], List[Dict[str, Any]]]:
    counts: Dict[str, int] = {
        "total_examples": 0,
        "correct_final": 0,
        "correct_draft": 0,
        "certified": 0,
        "exec_a_success": 0,
        "exec_b_success": 0,
        "both_valid": 0,
        "agreement": 0,
        "harm": 0,
        "model_calls": 0,
        "draft_correct_final_correct": 0,
        "draft_correct_final_incorrect": 0,
        "draft_incorrect_final_correct": 0,
        "draft_incorrect_final_incorrect": 0,
    }
    records: List[Dict[str, Any]] = []
    latency_sum = 0.0
    rel_tol = float(getattr(cfg.method_params, "float_tol", 1e-6))
    tokenizer = model_wrapper.tokenizer
    max_label_tokens = int(getattr(cfg.training, "max_label_tokens", 64))

    total_effective = get_effective_examples(
        len(dataset),
        max_examples,
        cfg.training.max_batches,
        cfg.training.batch_size,
    )
    if total_effective == 0:
        return {}, counts, records

    grad_accum_steps = max(1, int(getattr(cfg.training, "gradient_accumulation_steps", 1)))
    trainable_params = [p for p in model_wrapper.model.parameters() if p.requires_grad]
    if update_model and optimizer is None:
        raise ValueError("Optimizer must be provided when update_model=True.")
    if update_model and not trainable_params:
        raise ValueError("No trainable parameters available for optimizer updates.")
    accum_steps = 0
    if update_model and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for idx in range(total_effective):
        example = dataset[idx]
        question = example["question"]
        gold_value = example["gold_value"]

        if idx == 0:
            assert isinstance(question, str) and len(question) > 0
            assert gold_value is not None
            prompt = f"Question: {question}\nAnswer:"
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"]
            label_ids = tokenizer(
                str(example.get("answer_text", "")),
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
            )["input_ids"]
            assert input_ids.dim() == 2 and label_ids.dim() == 2
            assert input_ids.shape[0] == label_ids.shape[0]
            assert label_ids.shape[1] > 0

        model_wrapper.model.eval()
        start_time = time.time()
        result = runner.run_example(question, cfg.method)
        latency = time.time() - start_time
        latency_sum += latency

        pred_num = parse_number_from_text(result["final_answer"])
        draft_num = parse_number_from_text(result["draft_answer"])
        correct_final = numbers_match(pred_num, gold_value, rel_tol=rel_tol)
        correct_draft = numbers_match(draft_num, gold_value, rel_tol=rel_tol)

        counts["total_examples"] += 1
        counts["correct_final"] += int(correct_final)
        counts["correct_draft"] += int(correct_draft)
        counts["certified"] += int(result["certified"])
        counts["exec_a_success"] += int(result["exec_a_success"])
        counts["exec_b_success"] += int(result["exec_b_success"])
        counts["model_calls"] += int(result["model_calls"])

        both_valid = result["valid_a"] and result["valid_b"]
        counts["both_valid"] += int(both_valid)
        counts["agreement"] += int(both_valid and result["agreement"])

        harm = correct_draft and not correct_final
        counts["harm"] += int(harm)

        if correct_draft and correct_final:
            counts["draft_correct_final_correct"] += 1
        elif correct_draft and not correct_final:
            counts["draft_correct_final_incorrect"] += 1
        elif not correct_draft and correct_final:
            counts["draft_incorrect_final_correct"] += 1
        else:
            counts["draft_incorrect_final_incorrect"] += 1

        loss_value = None
        grad_norm = None
        if update_model and optimizer is not None and example.get("answer_text"):
            model_wrapper.model.train()
            loss = compute_supervised_loss(
                model_wrapper,
                question,
                example["answer_text"],
                max_label_tokens=max_label_tokens,
            )
            loss_value = float(loss.item())
            scaled_loss = loss / grad_accum_steps
            aux_grads = torch.autograd.grad(
                scaled_loss,
                trainable_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            aux_norms = [g.norm() for g in aux_grads if g is not None]
            grad_norm = float(torch.norm(torch.stack(aux_norms)).item()) if aux_norms else 0.0
            scaled_loss.backward()
            accum_steps += 1
            is_last = idx == total_effective - 1
            if accum_steps % grad_accum_steps == 0 or is_last:
                grad_values = [p.grad for p in trainable_params]
                assert all(g is not None for g in grad_values), "Gradients missing before optimizer step"
                total_grad = float(sum(g.abs().sum().item() for g in grad_values))
                assert total_grad > 0.0, "Gradients are zero before optimizer step"
                if cfg.training.max_grad_norm and cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            model_wrapper.model.eval()

        total = counts["total_examples"]
        accuracy = counts["correct_final"] / total
        draft_accuracy = counts["correct_draft"] / total
        certified_coverage = counts["certified"] / total
        exec_rate_a = counts["exec_a_success"] / total
        exec_rate_b = counts["exec_b_success"] / total
        agreement_rate = counts["agreement"] / counts["both_valid"] if counts["both_valid"] > 0 else 0.0
        harm_rate = counts["harm"] / total
        avg_latency = latency_sum / total
        avg_calls = counts["model_calls"] / total

        if wandb_run is not None and log_steps:
            metrics_log = {
                "accuracy": accuracy,
                "draft_accuracy": draft_accuracy,
                "certified_coverage": certified_coverage,
                "agreement_rate_when_both_valid": agreement_rate,
                "exec_rate_a": exec_rate_a,
                "exec_rate_b": exec_rate_b,
                "harm_rate": harm_rate,
                "latency_sec": latency,
                "avg_latency_sec": avg_latency,
                "model_calls": result["model_calls"],
                "avg_model_calls": avg_calls,
            }
            if loss_value is not None:
                metrics_log["train_loss"] = loss_value
            if grad_norm is not None:
                metrics_log["grad_norm"] = grad_norm
            if optimizer is not None:
                metrics_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(metrics_log, step=step_offset + idx)

        records.append(
            {
                "id": example["id"],
                "question": question,
                "gold": example["answer_text"],
                "gold_value": gold_value,
                "draft_answer": result["draft_answer"],
                "final_answer": result["final_answer"],
                "certified": result["certified"],
                "exec_a_status": result["exec_a_status"],
                "exec_b_status": result["exec_b_status"],
                "valid_a": result["valid_a"],
                "valid_b": result["valid_b"],
                "agreement": result["agreement"],
                "repair_used": result["repair_used"],
                "failure_reason": result["failure_reason"],
            }
        )

    total = counts["total_examples"]
    final_metrics = {
        "accuracy": counts["correct_final"] / total if total else 0.0,
        "draft_accuracy": counts["correct_draft"] / total if total else 0.0,
        "certified_coverage": counts["certified"] / total if total else 0.0,
        "agreement_rate_when_both_valid": counts["agreement"] / counts["both_valid"] if counts["both_valid"] > 0 else 0.0,
        "exec_rate_a": counts["exec_a_success"] / total if total else 0.0,
        "exec_rate_b": counts["exec_b_success"] / total if total else 0.0,
        "harm_rate": counts["harm"] / total if total else 0.0,
        "avg_latency_sec": latency_sum / total if total else 0.0,
        "avg_model_calls": counts["model_calls"] / total if total else 0.0,
        "total_examples": total,
    }
    return final_metrics, counts, records


def init_wandb(cfg: DictConfig) -> Any:
    if cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run.run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        mode=cfg.wandb.mode,
    )
    print(f"W&B run URL: {run.get_url()}")
    return run


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_mode(cfg)
    cfg = sync_timeout_cfg(cfg)
    ensure_run_node(cfg)
    assert cfg.run.run_id == cfg.run_id
    set_seed(cfg.training.seed)

    dataset = prepare_dataset(cfg)
    if not dataset:
        raise ValueError("Dataset is empty after preprocessing.")
    if cfg.training.batch_size != 1:
        raise ValueError("This experiment pipeline currently supports batch_size=1 for deterministic decoding.")

    model_wrapper = MathSolverModel(cfg.model)
    assert model_wrapper.tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set."
    assert model_wrapper.model.config.vocab_size > 0, "Model vocab size must be valid."
    output_embeddings = model_wrapper.model.get_output_embeddings()
    assert output_embeddings is not None
    assert output_embeddings.weight.shape[0] == model_wrapper.model.config.vocab_size

    cfg = run_optuna(cfg, dataset, model_wrapper)
    wandb_run = init_wandb(cfg)

    runner = MethodRunner(model_wrapper, cfg)
    update_model = bool(getattr(cfg.training, "update_model", False))
    optimizer = None
    scheduler = None
    grad_accum_steps = max(1, int(getattr(cfg.training, "gradient_accumulation_steps", 1)))

    if update_model:
        optimizer = create_optimizer(cfg, [p for p in model_wrapper.model.parameters() if p.requires_grad])
        effective_len = get_effective_examples(
            len(dataset),
            None,
            cfg.training.max_batches,
            cfg.training.batch_size,
        )
        steps_per_epoch = math.ceil(effective_len / grad_accum_steps)
        total_steps = max(1, steps_per_epoch * cfg.training.epochs)
        scheduler = create_scheduler(optimizer, cfg, total_steps)

    best_metrics: Dict[str, float] = {}
    best_counts: Dict[str, int] = {}
    best_accuracy = -1.0
    step_offset = 0
    all_records: List[Dict[str, Any]] = []

    for _epoch in range(cfg.training.epochs):
        metrics, counts, records = evaluate_dataset(
            cfg,
            dataset,
            model_wrapper,
            runner,
            wandb_run=wandb_run,
            max_examples=None,
            step_offset=step_offset,
            update_model=update_model,
            optimizer=optimizer,
            scheduler=scheduler,
            log_steps=True,
        )
        step_offset += counts["total_examples"]
        all_records.extend(records)
        if metrics.get("accuracy", 0.0) > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = metrics
            best_counts = counts

    if wandb_run is not None:
        if all_records:
            table_columns = list(all_records[0].keys())
            table = wandb.Table(data=[[row[col] for col in table_columns] for row in all_records], columns=table_columns)
            wandb.log({"examples": table})
        for key, value in best_metrics.items():
            wandb.summary[key] = value
        for key, value in best_counts.items():
            wandb.summary[key] = value
        wandb.finish()


if __name__ == "__main__":
    main()
