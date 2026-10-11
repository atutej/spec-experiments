#!/usr/bin/env python
"""Upload a finished SFT model (the root of $WORK_DIR/model, HF format) to the Hugging Face Hub, with a generated model card.

Uploads only the files of the final model: weights, config, tokenizer, chat template, result json files and the card. NOT the
checkpoint-N/ directories (optimizer states, several GB each) and NOT training_args.bin (a pickle). The repo is created if missing,
PRIVATE unless --public. Needs a Hugging Face token with write access (`hf auth login` in the `sft` env, or HF_TOKEN in the environment).

  python upload_to_hf.py --model-dir <WORK_DIR>/model --repo-id <namespace>/<name> [--public] [--dry-run]

--dry-run prints the files and the card and touches no network. The card is built from <model-dir>/../train_config.yaml and the result
files; read it before making a repo public.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

FILES = ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "train_results.json", "eval_results.json", "all_results.json"]


def read_yaml_flat(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#")[0].rstrip()
            if line and not line.startswith(" ") and ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def model_card(model_dir: Path, repo_id: str) -> str:
    cfg = read_yaml_flat(model_dir.parent / "train_config.yaml")
    res = {}
    for name in ("train_results.json", "eval_results.json"):
        p = model_dir / name
        if p.exists():
            res.update(json.loads(p.read_text()))
    state = json.loads((model_dir / "trainer_state.json").read_text()) if (model_dir / "trainer_state.json").exists() else {}
    steps = state.get("global_step")
    bsz = None
    if "per_device_train_batch_size" in cfg and "gradient_accumulation_steps" in cfg:
        bsz = f"{cfg['per_device_train_batch_size']} per device x {cfg['gradient_accumulation_steps']} accumulation x 4 GPUs"
    lines = [
        "---", "base_model: Qwen/Qwen3-0.6B", "datasets:", "- nvidia/Nemotron-Terminal-Corpus", "library_name: transformers",
        "pipeline_tag: text-generation", "tags:", "- sft", "- terminal-agent", "---", "",
        f"# {repo_id.split('/')[-1]}", "",
        "Full fine-tune of [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on the terminal-agent trajectories of "
        "[nvidia/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) (`dataset_adapters`; "
        "DeepSeek-V3.2 terminus-2 trajectories with thinking).", "",
        "## Training data", "",
        "A seeded random sample of 100,000 conversations (seed 0). Each conversation is expanded into one example per assistant turn: "
        "the messages up to and including that turn, with the loss on that turn's thinking and response only. Earlier turns are "
        "context, and their thinking is stripped, as the Qwen3 chat template does at inference. Examples longer than 16,384 tokens are "
        "dropped (not truncated).", "",
        "## Training", "",
        "- Trainer: LLaMA-Factory (upstream, commit `ce9dc9e0`), template `qwen3`, `mask_history`, full fine-tuning, fp32 master weights with "
        "bf16 autocast, Liger kernels, batches grouped by length",
        f"- Learning rate {cfg.get('learning_rate', '?')}, cosine schedule, warmup ratio {cfg.get('warmup_ratio', '?')}, "
        f"{cfg.get('num_train_epochs', '?')} epoch(s)" + (f", {steps} steps" if steps else ""),
    ]
    if bsz:
        lines.append(f"- Batch size: {bsz}")
    if "train_loss" in res or "eval_loss" in res:
        lines += ["", "## Results", ""]
        if "train_loss" in res:
            lines.append(f"- Mean training loss: {res['train_loss']:.4f}")
        if "eval_loss" in res:
            lines.append(f"- Validation loss (1,000 random held-out per-turn examples, not split by conversation, so somewhat "
                         f"optimistic): {res['eval_loss']:.4f}")
    lines += ["", "## Notes", "",
              "- The weights are saved in float32; load with `torch_dtype=torch.bfloat16` (vLLM: `--dtype bfloat16`) to match Qwen3's usual precision.",
              "- Base model: Qwen3-0.6B (Apache-2.0). Training data: Nemotron-Terminal-Corpus (CC-BY-4.0). Check both licenses before reuse.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo-id", required=True, help="<namespace>/<name>, or just <name> for the logged-in account")
    ap.add_argument("--public", action="store_true", help="make the repo public if it has to be created (default: private)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit-message", default="Upload SFT model")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    present = [f for f in FILES if (model_dir / f).exists()]
    missing = [f for f in ("model.safetensors", "config.json", "tokenizer.json") if f not in present]
    if missing:
        sys.exit(f"{model_dir} is not a finished model: missing {missing}")
    repo_id = args.repo_id if "/" in args.repo_id else f"<logged-in account>/{args.repo_id}"
    card = model_card(model_dir, repo_id)
    print(f"model dir: {model_dir}\nrepo: {repo_id} ({'public' if args.public else 'private'} if created)")
    for f in present:
        print(f"  {f:28s} {(model_dir / f).stat().st_size / 1e6:10.1f} MB")
    print("  README.md                    (generated model card)\n--- card ---\n" + card + "--- end card ---")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"no usable Hugging Face login ({type(e).__name__}); run `hf auth login` in the sft env or set HF_TOKEN")
    print(f"logged in as {who}")
    if "/" not in args.repo_id:
        args.repo_id = f"{who}/{args.repo_id}"
        card = model_card(model_dir, args.repo_id)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in present:
            (Path(tmp) / f).symlink_to((model_dir / f).resolve())
        (Path(tmp) / "README.md").write_text(card)
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id, repo_type="model", commit_message=args.commit_message)
    info = api.model_info(args.repo_id)
    print(f"uploaded: https://huggingface.co/{args.repo_id} (private={info.private}, {len(info.siblings or [])} files)")


if __name__ == "__main__":
    main()
