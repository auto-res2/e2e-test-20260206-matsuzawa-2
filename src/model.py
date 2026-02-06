import ast
import builtins
import math
import multiprocessing as mp
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


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


def extract_number(text: Any) -> Optional[str]:
    if text is None:
        return None
    text = str(text).replace(",", "")
    matches = NUM_RE.findall(text)
    return matches[-1] if matches else None


def format_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def normalize_exec_result(result: Any) -> Optional[float]:
    if isinstance(result, bool) or result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result) if math.isfinite(result) else None
    try:
        value = float(str(result).strip())
        return value if math.isfinite(value) else None
    except Exception:
        return None


def same_number(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def build_prompt(tokenizer: AutoTokenizer, messages: list[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = ""
    for msg in messages:
        role = msg["role"].capitalize()
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant:"
    return prompt


def render_template(template: str, **kwargs: Any) -> str:
    text = str(template)
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))
    return text


def ensure_placeholders(template: str, placeholders: list[str], name: str) -> None:
    for placeholder in placeholders:
        token = "{" + placeholder + "}"
        if token not in template:
            raise ValueError(f"Prompt template '{name}' missing placeholder {token}.")


def get_prompt_value(prompts: Any, key: str) -> Optional[str]:
    if prompts is None:
        return None
    if isinstance(prompts, dict):
        return prompts.get(key)
    if hasattr(prompts, key):
        return getattr(prompts, key)
    return None


def validate_positive_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be provided in the configuration.")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value}.") from exc
    if value_int <= 0:
        raise ValueError(f"{name} must be positive, got {value_int}.")
    return value_int


def _sandbox_worker(code_str: str, allowed_builtins: list[str], queue: mp.Queue) -> None:
    try:
        safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
        env = {"__builtins__": safe_builtins, "math": math}
        loc: Dict[str, Any] = {}
        exec(code_str, env, loc)
        if "solve" not in loc or not callable(loc["solve"]):
            raise ValueError("No solve() function defined")
        result = loc["solve"]()
        queue.put(("ok", result, None))
    except Exception as exc:
        queue.put(("err", None, repr(exc)))


class SandboxExecutor:
    def __init__(self, allowed_builtins: list[str], timeout_s: float) -> None:
        self.allowed_builtins = list(allowed_builtins)
        self.timeout_s = float(timeout_s)

    def run_code_safely(self, code_str: str) -> Tuple[str, Optional[Any], Optional[str]]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_sandbox_worker, args=(code_str, self.allowed_builtins, queue))
        proc.start()
        proc.join(self.timeout_s)
        if proc.is_alive():
            proc.terminate()
            return "timeout", None, "Timeout"
        if queue.empty():
            return "err", None, "No output"
        return queue.get()


class GroundednessChecker:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def check(self, code_str: str, question: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return False, "syntax"
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
            return False, "import"
        has_solve = any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in ast.walk(tree))
        if not has_solve:
            return False, "no_solve"
        has_op = any(isinstance(n, (ast.BinOp, ast.UnaryOp)) for n in ast.walk(tree))
        q_nums = set(NUM_RE.findall(str(question).replace(",", "")))
        consts = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                consts.append(str(n.value))
        uses_q_num = any(c in q_nums or c.rstrip(".0") in q_nums for c in consts)
        question_tokens = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(question).lower()))
        code_names = set(n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name))
        name_overlap = bool(question_tokens & code_names)

        if "has_op_only" in self.mode:
            return (has_op, "ok" if has_op else "no_op")
        if "literal_or_varname_overlap" in self.mode:
            if not has_op:
                return False, "no_op"
            if q_nums and not (uses_q_num or name_overlap):
                return False, "no_literal_or_var_overlap"
            return True, "ok"

        if not has_op:
            return False, "no_op"
        if q_nums and not uses_q_num:
            return False, "no_question_number_used"
        return True, "ok"


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str is None:
        return torch.float32
    s = str(dtype_str).lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


class MathSolverModel:
    def __init__(self, model_cfg: Any) -> None:
        self.model_name = model_cfg.name
        device = model_cfg.device
        dtype = resolve_dtype(model_cfg.dtype)
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        self.decoding = str(getattr(model_cfg, "decoding", "greedy")).lower()
        self.temperature = float(getattr(model_cfg, "temperature", 1.0))
        self.top_p = float(getattr(model_cfg, "top_p", 1.0))
        self.top_k = max(0, int(getattr(model_cfg, "top_k", 50)))
        self.num_beams = max(1, int(getattr(model_cfg, "num_beams", 1)))

        cache_dir = ensure_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if device == "cpu":
            self.model.to("cpu")
        if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = next(self.model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id must be set.")
        if self.model.config.vocab_size <= 0:
            raise ValueError("Model vocabulary size is invalid.")

    def _generation_kwargs(self, max_new_tokens: int) -> Dict[str, Any]:
        decoding = self.decoding
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if decoding in {"greedy", "deterministic"}:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
        elif decoding in {"sample", "sampling"}:
            kwargs["do_sample"] = True
            kwargs["temperature"] = max(self.temperature, 1e-5)
            kwargs["top_p"] = min(max(self.top_p, 0.0), 1.0)
            kwargs["top_k"] = self.top_k
        elif decoding in {"beam", "beam_search"}:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = self.num_beams
        else:
            raise ValueError(f"Unsupported decoding strategy: {decoding}")
        return kwargs

    def generate_text(self, messages: list[Dict[str, str]], max_new_tokens: int) -> str:
        prompt = build_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if max_pos is not None:
            max_new_tokens = min(max_new_tokens, max_pos - input_len - 1)
            max_new_tokens = max(1, max_new_tokens)
        gen_kwargs = self._generation_kwargs(max_new_tokens)
        with torch.no_grad():
            ctx = torch.autocast(device_type="cuda", dtype=self.model.dtype) if self.device.type == "cuda" else nullcontext()
            with ctx:
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
        generated = output_ids[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class MethodRunner:
    def __init__(self, model: MathSolverModel, cfg: Any) -> None:
        self.model = model
        self.cfg = cfg
        sandbox_cfg = getattr(cfg.method_params, "sandbox", None)
        allowed_builtins = getattr(sandbox_cfg, "allowed_builtins", None) if sandbox_cfg else None
        if not allowed_builtins:
            allowed_builtins = ["abs", "min", "max", "sum", "round"]
        timeout_s = getattr(cfg.method_params, "exec_timeout_s", None)
        if timeout_s is None:
            timeout_s = getattr(sandbox_cfg, "timeout_s", 1.0) if sandbox_cfg else 1.0
        self.executor = SandboxExecutor(allowed_builtins, float(timeout_s))
        mode = getattr(cfg.method_params, "groundedness_mode", "literal_from_question+has_op (default)")
        self.groundedness = GroundednessChecker(mode)
        self.float_tol = float(getattr(cfg.method_params, "float_tol", 1e-6))

        self.prompts = getattr(cfg, "prompts", None)
        self.system_prompt = self._get_prompt("system", allow_empty=True)
        self.prompt_zero_shot = self._get_prompt("zero_shot")
        self.prompt_zero_shot_cot = self._get_prompt("zero_shot_cot")
        self.prompt_zero_shot_answer = self._get_prompt("zero_shot_answer")
        self.prompt_mv_cot = self._get_prompt("mv_cot")
        self.prompt_compile_rationale = self._get_prompt("compile_rationale")
        self.prompt_compile_question = self._get_prompt("compile_question")
        self.prompt_repair = self._get_prompt("repair")
        self._validate_prompts()

        self.max_new_tokens_zs = validate_positive_int(
            getattr(cfg.model, "max_new_tokens_zs", None),
            "model.max_new_tokens_zs",
        )
        self.max_new_tokens_answer = validate_positive_int(
            getattr(cfg.model, "max_new_tokens_answer", None),
            "model.max_new_tokens_answer",
        )
        self.max_new_tokens_mv = validate_positive_int(
            getattr(cfg.model, "max_new_tokens_mv", None),
            "model.max_new_tokens_mv",
        )
        self.max_new_tokens_cot = validate_positive_int(
            getattr(cfg.model, "max_new_tokens_cot", None),
            "model.max_new_tokens_cot",
        )
        self.max_new_tokens_code = validate_positive_int(
            getattr(cfg.model, "max_new_tokens_code", None),
            "model.max_new_tokens_code",
        )
        repair_tokens = getattr(cfg.model, "max_new_tokens_repair", None)
        if repair_tokens is None:
            repair_tokens = self.max_new_tokens_code
        self.max_new_tokens_repair = validate_positive_int(repair_tokens, "model.max_new_tokens_repair")

        self.use_repair_round = bool(getattr(cfg.method_params, "use_repair_round", True))
        self.call_count = 0

    def _get_prompt(self, key: str, allow_empty: bool = False) -> str:
        value = get_prompt_value(self.prompts, key)
        if value is None:
            raise ValueError(f"Missing prompts.{key} in configuration.")
        text = str(value)
        if not allow_empty and text.strip() == "":
            raise ValueError(f"Prompt prompts.{key} must be non-empty.")
        return text

    def _validate_prompts(self) -> None:
        ensure_placeholders(self.prompt_zero_shot, ["question"], "zero_shot")
        ensure_placeholders(self.prompt_zero_shot_cot, ["question"], "zero_shot_cot")
        ensure_placeholders(self.prompt_zero_shot_answer, ["question", "rationale"], "zero_shot_answer")
        ensure_placeholders(self.prompt_mv_cot, ["question", "rationale"], "mv_cot")
        ensure_placeholders(self.prompt_compile_rationale, ["question", "rationale"], "compile_rationale")
        ensure_placeholders(self.prompt_compile_question, ["question"], "compile_question")
        ensure_placeholders(
            self.prompt_repair,
            ["question", "why_fail", "code_a", "out_a", "err_a", "code_b", "out_b", "err_b"],
            "repair",
        )

    def _build_messages(self, user_content: str) -> list[Dict[str, str]]:
        messages: list[Dict[str, str]] = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate(self, messages: list[Dict[str, str]], max_new_tokens: int) -> str:
        self.call_count += 1
        return self.model.generate_text(messages, max_new_tokens)

    def _extract_code(self, text: str) -> str:
        match = CODE_FENCE_RE.search(text or "")
        code = match.group(1).strip() if match else (text or "").strip()
        if "def solve" not in code:
            num = extract_number(code)
            fallback = num if num is not None else "None"
            code = f"def solve():\n    return {fallback}\n"
        return code

    def zeroshot(self, question: str) -> str:
        content = render_template(self.prompt_zero_shot, question=question)
        messages = self._build_messages(content)
        out = self._generate(messages, max_new_tokens=self.max_new_tokens_zs)
        ans = extract_number(out)
        return ans or ""

    def zeroshot_cot(self, question: str) -> Tuple[str, str]:
        content = render_template(self.prompt_zero_shot_cot, question=question)
        messages = self._build_messages(content)
        rationale = self._generate(messages, max_new_tokens=self.max_new_tokens_cot)
        answer_prompt = render_template(self.prompt_zero_shot_answer, question=question, rationale=rationale)
        ans_messages = self._build_messages(answer_prompt)
        ans_text = self._generate(ans_messages, max_new_tokens=self.max_new_tokens_answer)
        ans = extract_number(ans_text)
        return (ans or ""), rationale

    def mv_cot(self, question: str) -> Dict[str, Any]:
        draft_ans, rationale = self.zeroshot_cot(question)
        content = render_template(self.prompt_mv_cot, question=question, rationale=rationale)
        messages = self._build_messages(content)
        out = self._generate(messages, max_new_tokens=self.max_new_tokens_mv)
        ans = extract_number(out) or draft_ans
        return {
            "draft_answer": draft_ans,
            "final_answer": ans,
            "rationale": rationale,
            "failure_reason": "",
        }

    def synth_code_from_rationale(self, question: str, rationale: str) -> str:
        content = render_template(self.prompt_compile_rationale, question=question, rationale=rationale)
        messages = self._build_messages(content)
        return self._extract_code(self._generate(messages, max_new_tokens=self.max_new_tokens_code))

    def synth_code_from_question(self, question: str) -> str:
        content = render_template(self.prompt_compile_question, question=question)
        messages = self._build_messages(content)
        return self._extract_code(self._generate(messages, max_new_tokens=self.max_new_tokens_code))

    def repair_code(
        self,
        question: str,
        code_a: str,
        out_a: Any,
        err_a: Optional[str],
        code_b: str,
        out_b: Any,
        err_b: Optional[str],
        why_fail: str,
    ) -> str:
        content = render_template(
            self.prompt_repair,
            question=question,
            why_fail=why_fail,
            code_a=code_a,
            out_a=out_a,
            err_a=err_a,
            code_b=code_b,
            out_b=out_b,
            err_b=err_b,
        )
        messages = self._build_messages(content)
        return self._extract_code(self._generate(messages, max_new_tokens=self.max_new_tokens_repair))

    def emv_cot(self, question: str) -> Dict[str, Any]:
        draft_ans, rationale = self.zeroshot_cot(question)
        code_a = self.synth_code_from_rationale(question, rationale)
        st_a, res_a, err_a = self.executor.run_code_safely(code_a)
        exec_a_success = st_a == "ok" and normalize_exec_result(res_a) is not None
        final_answer = draft_ans
        certified = False
        repair_used = False
        failure_reason = ""
        if exec_a_success:
            final_answer = format_number(normalize_exec_result(res_a))
            certified = True
        elif self.use_repair_round:
            repair_used = True
            patched_a = self.repair_code(question, code_a, res_a, err_a, "", None, None, f"exec_failed ({st_a})")
            st_p, res_p, err_p = self.executor.run_code_safely(patched_a)
            exec_p = st_p == "ok" and normalize_exec_result(res_p) is not None
            if exec_p:
                final_answer = format_number(normalize_exec_result(res_p))
                certified = True
                code_a = patched_a
                st_a, res_a, err_a = st_p, res_p, err_p
                exec_a_success = True
            else:
                failure_reason = f"repair_failed ({st_p})"
        else:
            failure_reason = f"exec_failed ({st_a})"

        return {
            "draft_answer": draft_ans,
            "final_answer": final_answer,
            "rationale": rationale,
            "code_a": code_a,
            "code_b": "",
            "exec_a_status": st_a,
            "exec_b_status": "",
            "exec_a_success": exec_a_success,
            "exec_b_success": False,
            "valid_a": exec_a_success,
            "valid_b": False,
            "agreement": False,
            "certified": certified,
            "repair_used": repair_used,
            "failure_reason": failure_reason,
        }

    def dc_emv_cot(self, question: str) -> Dict[str, Any]:
        draft_ans, rationale = self.zeroshot_cot(question)
        code_a = self.synth_code_from_rationale(question, rationale)
        code_b = self.synth_code_from_question(question)

        ok_a, why_a = self.groundedness.check(code_a, question)
        ok_b, why_b = self.groundedness.check(code_b, question)

        st_a, res_a, err_a = self.executor.run_code_safely(code_a)
        st_b, res_b, err_b = self.executor.run_code_safely(code_b)

        exec_a_success = st_a == "ok" and normalize_exec_result(res_a) is not None
        exec_b_success = st_b == "ok" and normalize_exec_result(res_b) is not None

        valid_a = exec_a_success and ok_a
        valid_b = exec_b_success and ok_b

        va = normalize_exec_result(res_a) if valid_a else None
        vb = normalize_exec_result(res_b) if valid_b else None

        agree = valid_a and valid_b and same_number(va, vb, tol=self.float_tol)
        certified = False
        final_answer = draft_ans
        repair_used = False
        failure_reason = ""

        if agree:
            certified = True
            final_answer = format_number(va)
        elif self.use_repair_round:
            repair_used = True
            why_fail = f"agree_failed (A:{st_a}/{why_a} B:{st_b}/{why_b})"
            patched_a = self.repair_code(question, code_a, res_a, err_a, code_b, res_b, err_b, why_fail)
            ok_p, why_p = self.groundedness.check(patched_a, question)
            st_p, res_p, err_p = self.executor.run_code_safely(patched_a)
            exec_p = st_p == "ok" and normalize_exec_result(res_p) is not None
            valid_p = exec_p and ok_p
            vp = normalize_exec_result(res_p) if valid_p else None

            if valid_p and valid_b and same_number(vp, vb, tol=self.float_tol):
                certified = True
                final_answer = format_number(vp)
                code_a = patched_a
                st_a, res_a, err_a = st_p, res_p, err_p
                exec_a_success = exec_p
                valid_a = valid_p
                agree = True
            elif valid_p and valid_a and same_number(vp, va, tol=self.float_tol):
                certified = True
                final_answer = format_number(vp)
                code_a = patched_a
                st_a, res_a, err_a = st_p, res_p, err_p
                exec_a_success = exec_p
                valid_a = valid_p
                agree = True
            else:
                failure_reason = f"repair_failed ({st_p}/{why_p})"
        else:
            failure_reason = "agreement_or_exec_failed"

        return {
            "draft_answer": draft_ans,
            "final_answer": final_answer,
            "rationale": rationale,
            "code_a": code_a,
            "code_b": code_b,
            "exec_a_status": st_a,
            "exec_b_status": st_b,
            "exec_a_success": exec_a_success,
            "exec_b_success": exec_b_success,
            "valid_a": valid_a,
            "valid_b": valid_b,
            "agreement": agree,
            "certified": certified,
            "repair_used": repair_used,
            "failure_reason": failure_reason,
        }

    def run_example(self, question: str, method_name: str) -> Dict[str, Any]:
        self.call_count = 0
        method = method_name or ""
        if "DC-eMV-CoT" in method:
            result = self.dc_emv_cot(question)
        elif "eMV-CoT" in method:
            result = self.emv_cot(question)
        elif "MV-CoT" in method:
            result = self.mv_cot(question)
        elif "Zero-shot-CoT" in method:
            draft, rationale = self.zeroshot_cot(question)
            result = {
                "draft_answer": draft,
                "final_answer": draft,
                "rationale": rationale,
                "failure_reason": "",
            }
        else:
            ans = self.zeroshot(question)
            result = {
                "draft_answer": ans,
                "final_answer": ans,
                "rationale": "",
                "failure_reason": "",
            }

        defaults = {
            "draft_answer": "",
            "final_answer": "",
            "rationale": "",
            "code_a": "",
            "code_b": "",
            "exec_a_status": "",
            "exec_b_status": "",
            "exec_a_success": False,
            "exec_b_success": False,
            "valid_a": False,
            "valid_b": False,
            "agreement": False,
            "certified": False,
            "repair_used": False,
            "failure_reason": "",
        }
        output = {**defaults, **result}
        output["model_calls"] = self.call_count
        return output
