import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect module names/classes for non-Qwen multimodal models. "
            "Use this before designing model-specific audio-path gates."
        )
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--name_regex",
        default=(
            "audio|speech|sound|whisper|wav|mel|encoder|adapter|project|"
            "proj|connector|modality|multi|vision|lora|mm"
        ),
        help="Case-insensitive regex used to filter module names/classes.",
    )
    parser.add_argument(
        "--all_modules",
        action="store_true",
        help="Save every module, not only regex matches.",
    )
    parser.add_argument(
        "--loader",
        default="auto",
        choices=["auto", "causal_lm", "auto_model", "remote_class"],
        help="Model loader to use. auto tries AutoModelForCausalLM then AutoModel.",
    )
    parser.add_argument(
        "--remote_class",
        default=None,
        help=(
            "Remote class reference such as modeling_phi4mm.Phi4MMForCausalLM. "
            "Used with --loader remote_class or prepended to --loader auto."
        ),
    )
    parser.add_argument(
        "--patch_prepare_inputs_for_generation",
        action="store_true",
        help=(
            "Monkey-patch missing prepare_inputs_for_generation on custom remote "
            "classes. Useful for Phi-4-MM inspection with newer transformers."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help=(
            "Device map passed to from_pretrained. Use 'cpu' for CPU-only "
            "inspection or 'auto' on GPU servers."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Optional attention backend passed to from_pretrained. Use eager "
            "or sdpa when a model defaults to FlashAttention2 but flash_attn "
            "is unavailable."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use cached model files.",
    )
    parser.add_argument(
        "--max_rows_print",
        type=int,
        default=200,
        help="Print only the first N rows to stdout; JSON file always stores all rows.",
    )
    parser.add_argument(
        "--static_code_dir",
        default=None,
        help=(
            "Inspect cached remote-code .py files without instantiating the model. "
            "Use this when custom HF loaders fail before named_modules() is available."
        ),
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def direct_child_count(module: torch.nn.Module) -> int:
    return sum(1 for _ in module.children())


def direct_param_count(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters(recurse=False))


def force_attention_backend(config: Any, backend: str, visited: set[int] | None = None) -> None:
    if visited is None:
        visited = set()
    obj_id = id(config)
    if obj_id in visited:
        return
    visited.add(obj_id)

    for attr in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
        try:
            setattr(config, attr, backend)
        except Exception:
            pass

    if isinstance(config, dict):
        values = config.values()
    elif isinstance(config, (list, tuple)):
        values = config
    elif hasattr(config, "__dict__"):
        values = vars(config).values()
    else:
        return

    for value in values:
        if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"):
            force_attention_backend(value, backend, visited)


def patch_prepare_inputs_for_generation(cls: Any) -> bool:
    if hasattr(cls, "prepare_inputs_for_generation"):
        return False

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_inputs: Dict[str, Any]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        model_inputs.update(kwargs)
        return model_inputs

    setattr(cls, "prepare_inputs_for_generation", prepare_inputs_for_generation)
    return True


def load_remote_class(args: argparse.Namespace, class_reference: str) -> Any:
    cls = get_class_from_dynamic_module(
        class_reference,
        args.model_id,
        local_files_only=args.local_files_only,
    )
    if args.patch_prepare_inputs_for_generation:
        patch_prepare_inputs_for_generation(cls)
    return cls


def patch_known_remote_generation_classes(args: argparse.Namespace) -> List[str]:
    if not args.patch_prepare_inputs_for_generation:
        return []

    patched: List[str] = []
    class_refs = [
        args.remote_class,
        "modeling_phi4mm.Phi4MMModel",
        "modeling_phi4mm.Phi4MMForCausalLM",
    ]
    for class_ref in class_refs:
        if not class_ref:
            continue
        try:
            cls = load_remote_class(args, class_ref)
        except Exception:
            continue
        if patch_prepare_inputs_for_generation(cls):
            patched.append(class_ref)
    return patched


def load_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, str]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        force_attention_backend(config, args.attn_implementation)
        kwargs["config"] = config
        kwargs["attn_implementation"] = args.attn_implementation

    patched = patch_known_remote_generation_classes(args)
    errors: List[str] = []
    loaders = []
    if args.remote_class and args.loader in {"auto", "remote_class"}:
        try:
            remote_cls = load_remote_class(args, args.remote_class)
            loaders.append((f"RemoteClass:{args.remote_class}", remote_cls))
        except Exception as exc:
            errors.append(f"RemoteClass:{args.remote_class}: {type(exc).__name__}: {exc}")
    if args.loader in {"auto", "causal_lm"}:
        loaders.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    if args.loader in {"auto", "auto_model"}:
        loaders.append(("AutoModel", AutoModel))

    for label, loader in loaders:
        try:
            return loader.from_pretrained(args.model_id, **kwargs), label
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    missing = sorted(set(re.findall(r"No module named '([^']+)'", joined)))
    hint = ""
    if missing:
        hint = (
            "\n\nHint: this model's remote HF modeling code is missing "
            f"package(s): {', '.join(missing)}. Install them in the active "
            "environment, then rerun this command:\n"
            f"  python -m pip install {' '.join(missing)}\n"
            "Phi-4-MM cannot be inspected through the plain AutoModel fallback; "
            "its custom causal-LM loader must import the model code successfully."
        )
    if "FlashAttention2" in joined or "flash_attn" in joined:
        hint += (
            "\n\nHint: FlashAttention2 was requested but flash_attn is unavailable. "
            "For module inspection, rerun with --attn_implementation eager "
            "or --attn_implementation sdpa. Installing flash_attn is not "
            "necessary just to list module names."
        )
    patched_note = ""
    if patched:
        patched_note = "\nPatched remote classes: " + ", ".join(patched)
    raise RuntimeError("All loaders failed:\n" + joined + patched_note + hint)


def module_rows(
    model: torch.nn.Module,
    regex: str,
    all_modules: bool,
) -> List[Dict[str, Any]]:
    pattern = re.compile(regex, re.IGNORECASE)
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        text = f"{name} {class_name}"
        matched = bool(pattern.search(text))
        if not all_modules and not matched:
            continue
        child_count = direct_child_count(module)
        rows.append(
            {
                "name": name,
                "class_name": class_name,
                "matched": matched,
                "direct_child_count": child_count,
                "direct_param_count": direct_param_count(module),
                "is_leaf": child_count == 0,
            }
        )
    return rows


def static_code_rows(
    code_dir: str,
    regex: str,
    all_modules: bool,
) -> List[Dict[str, Any]]:
    pattern = re.compile(regex, re.IGNORECASE)
    rows: List[Dict[str, Any]] = []
    class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
    assign_re = re.compile(
        r"^\s*self\.([A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)?"
    )
    function_re = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)")

    for root, _, files in os.walk(code_dir):
        for file_name in sorted(files):
            if not file_name.endswith(".py"):
                continue
            path = os.path.join(root, file_name)
            rel_path = os.path.relpath(path, code_dir)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

            for line_no, raw_line in enumerate(lines, start=1):
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                kind = None
                name = None
                target = None
                class_match = class_re.match(line)
                assign_match = assign_re.match(line)
                function_match = function_re.match(line)
                if class_match:
                    kind = "class"
                    name = class_match.group(1)
                elif assign_match:
                    kind = "self_assignment"
                    name = assign_match.group(1)
                    target = assign_match.group(2)
                elif function_match:
                    kind = "function"
                    name = function_match.group(1)
                elif pattern.search(line):
                    kind = "matched_line"

                if kind is None:
                    continue

                text = " ".join(part for part in [rel_path, name or "", target or "", stripped] if part)
                matched = bool(pattern.search(text))
                if not all_modules and not matched:
                    continue
                rows.append(
                    {
                        "file": rel_path,
                        "line": line_no,
                        "kind": kind,
                        "name": name,
                        "target": target,
                        "matched": matched,
                        "text": stripped,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.static_code_dir:
        rows = static_code_rows(args.static_code_dir, args.name_regex, args.all_modules)
        output = {
            "model_id": args.model_id,
            "loader": "static_remote_code",
            "static_code_dir": args.static_code_dir,
            "name_regex": args.name_regex,
            "all_modules": args.all_modules,
            "num_rows": len(rows),
            "modules": rows,
        }
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        preview = dict(output)
        preview["modules"] = rows[: args.max_rows_print]
        preview["printed_rows"] = len(preview["modules"])
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return

    model, loader_name = load_model(args)
    rows = module_rows(model, args.name_regex, args.all_modules)
    output = {
        "model_id": args.model_id,
        "loader": loader_name,
        "name_regex": args.name_regex,
        "all_modules": args.all_modules,
        "num_rows": len(rows),
        "modules": rows,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    preview = dict(output)
    preview["modules"] = rows[: args.max_rows_print]
    preview["printed_rows"] = len(preview["modules"])
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
