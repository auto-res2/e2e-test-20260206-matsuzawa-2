import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_cache_dir() -> str:
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = str(get_repo_root() / ".cache")
        os.environ["HF_HOME"] = cache_dir
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("DATASETS_CACHE", cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def parse_number_from_text(text: Any) -> Optional[float]:
    if text is None:
        return None
    text = str(text).replace(",", "")
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_gold_answer(answer_text: Any, mode: Optional[str]) -> str:
    if answer_text is None:
        return ""
    answer_text = str(answer_text)
    if mode in (None, "", "none"):
        return answer_text.strip()
    if mode == "split_after_####":
        if "####" in answer_text:
            answer_text = answer_text.split("####")[-1]
        return answer_text.strip()
    raise ValueError(f"Unsupported answer_extraction mode: {mode}")


def numbers_match(pred: Optional[float], gold: Optional[float], rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    diff = abs(pred - gold)
    tol = max(abs_tol, rel_tol * abs(gold))
    if diff <= tol:
        return True
    if float(pred).is_integer() and float(gold).is_integer():
        return int(round(pred)) == int(round(gold))
    return False


def load_raw_dataset(cfg: DictConfig):
    cache_dir = ensure_cache_dir()
    if getattr(cfg, "config", None):
        return load_dataset(cfg.name, cfg.config, split=cfg.split, cache_dir=cache_dir)
    return load_dataset(cfg.name, split=cfg.split, cache_dir=cache_dir)


def preprocess_example(example: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    question = example[cfg.question_field]
    answer_text = example[cfg.answer_field]
    if getattr(cfg.preprocessing, "strip_whitespace", False):
        if isinstance(question, str):
            question = question.strip()
        if isinstance(answer_text, str):
            answer_text = answer_text.strip()
    gold_text = extract_gold_answer(answer_text, cfg.preprocessing.answer_extraction)
    gold_value = parse_number_from_text(gold_text)
    return {
        "question": question,
        "answer_text": gold_text,
        "gold_value": gold_value,
    }


def prepare_dataset(cfg: DictConfig) -> List[Dict[str, Any]]:
    ds_cfg = cfg.dataset if hasattr(cfg, "dataset") else cfg
    if not hasattr(ds_cfg, "question_field") or not hasattr(ds_cfg, "answer_field"):
        raise ValueError("Dataset configuration must include question_field and answer_field.")
    OmegaConf.set_struct(ds_cfg, False)
    if not hasattr(ds_cfg, "preprocessing") or ds_cfg.preprocessing is None:
        ds_cfg.preprocessing = OmegaConf.create({"strip_whitespace": True, "answer_extraction": None})
    if not hasattr(ds_cfg.preprocessing, "strip_whitespace"):
        ds_cfg.preprocessing.strip_whitespace = True
    if not hasattr(ds_cfg.preprocessing, "answer_extraction"):
        ds_cfg.preprocessing.answer_extraction = None
    raw = load_raw_dataset(ds_cfg)
    processed: List[Dict[str, Any]] = []
    for idx, example in enumerate(raw):
        item = preprocess_example(example, ds_cfg)
        item["id"] = idx
        processed.append(item)
    return processed
