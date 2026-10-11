#!/usr/bin/env python
"""Upload a finished speculator (drafter) run to the Hugging Face Hub: the best and the latest checkpoint, with a generated model card.

`--checkpoints-dir` is the trainer's save path ($WORK_DIR/checkpoints): numbered checkpoint directories (`0`, ...), links such as
`checkpoint_best` and `epoch0_end`, and run.yaml. This picks
  best   = `checkpoint_best` (chosen by validation; absent if validation never ran)
  latest = the checkpoint with the highest (epoch, global step)
and uploads them as ONE repo: the root holds the best checkpoint (the latest if there is no best), and `latest/` holds the latest one when
it differs from the best (a one-epoch run has a single checkpoint: best = latest, nothing is duplicated).
Per checkpoint only: model.safetensors, config.json, config.py (the custom config class `auto_map` points to), val_metrics.json,
training_state.json. NOT the optimizer and scheduler state (about as large as the weights) and not train_command.txt (local paths).

  python upload_drafter_to_hf.py --checkpoints-dir <WORK_DIR>/checkpoints --repo-id <namespace>/<name> --run-name NAME \
      [--description TEXT] [--public] [--dry-run]

Needs a write token (HF_TOKEN, or `hf auth login`). The repo is created private unless --public. --dry-run touches no network.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

FILES = ["model.safetensors", "config.json", "config.py", "val_metrics.json", "training_state.json"]


def state_key(d: Path) -> tuple[int, int]:
    p = d / "training_state.json"
    if not p.exists():
        return (-1, -1)
    s = json.loads(p.read_text())
    return (int(s.get("epoch", -1)), int(s.get("global_step", -1)))


def pick(root: Path) -> tuple[Path | None, Path]:
    cands = {d.resolve() for d in root.iterdir() if d.is_dir() and (d / "model.safetensors").exists()}
    if not cands:
        sys.exit(f"no checkpoint with model.safetensors in {root}")
    latest = max(cands, key=state_key)
    best = (root / "checkpoint_best").resolve() if (root / "checkpoint_best").exists() else None
    return best, latest


def yaml_flat(path: Path) -> dict:
    """Flatten the simple nested run.yaml (scalars and lists of scalars) to dotted keys."""
    out, stack = {}, []
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = (len(raw) - len(raw.lstrip())) // 2
        line = raw.strip()
        if line.startswith("- "):
            key = ".".join(stack[: indent]) if stack else ""
            out.setdefault(key, []).append(line[2:])
            continue
        k, _, v = line.partition(":")
        stack = stack[:indent] + [k]
        if v.strip():
            out[".".join(stack)] = v.strip()
    return out


def card(root: Path, best: Path | None, latest: Path, repo_id: str, run_name: str, description: str) -> str:
    y = yaml_flat(root / "run.yaml")
    shown = best or latest
    cfg = json.loads((shown / "config.json").read_text())
    verifier = cfg.get("speculators_config", {}).get("verifier", {}).get("name_or_path") or y.get("train.verifier.verifier_name_or_path", "?")
    val = json.loads((shown / "val_metrics.json").read_text()) if (shown / "val_metrics.json").exists() else None
    hub_id = verifier if "/" in verifier and not verifier.startswith("/") else None
    fm = ["---"] + ([f"base_model: {hub_id}"] if hub_id else []) + ["datasets:", "- nvidia/Nemotron-Terminal-Corpus", "tags:",
                                                                      "- speculative-decoding", "- dspark", "- drafter", "---", ""]
    L = fm + [f"# {repo_id.split('/')[-1]}", "",
              f"A DSpark speculative-decoding drafter (block size {cfg.get('block_size')}, {y.get('train.draft.num_layers', '?')} layers, reduced "
              f"draft vocabulary {cfg.get('draft_vocab_size')}) for the target model **{verifier}**. Trained with the `speculators` library "
              "(fork `atutej/speculators`, branch `nemotron-terminal-preset`); run name `" + run_name + "`.", ""]
    if description:
        L += [description, ""]
    L += ["## Training", "",
          f"- Online training: hidden states of layers {y.get('train.draft.target_layer_ids', '?')} of the target, streamed by vLLM through Mooncake",
          f"- Sequence length {y.get('train.data.total_seq_len', '?')}, max anchors {y.get('train.data.max_anchors', '?')}, learning rate "
          f"{y.get('train.optimizer.lr', '?')}, {y.get('train.trainer.epochs', '?')} epoch(s), loss {y.get('train.loss.loss_fn', '?')}",
          f"- Markov head rank {y.get('train.dspark.markov_rank', '?')} ({y.get('train.dspark.markov_head_type', '?')}), confidence head "
          f"{y.get('train.dspark.enable_confidence_head', '?')}"]
    if val:
        L += ["", "## Validation (teacher-forced, on rows of the training distribution)", ""]
        for k, v in val.items():
            L.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    L += ["", "## Checkpoints", ""]
    if best is None:
        L.append("The root is the latest checkpoint (no validation-selected best checkpoint exists for this run).")
    elif best == latest:
        L.append("A single checkpoint: the best (by validation) is also the latest, at the repo root.")
    else:
        L.append("The root is the best checkpoint (by validation); `latest/` is the latest one.")
    L += ["", "## Use", "",
          "With the Marin vLLM fork (supports `method: dspark`):", "",
          "```", f"vllm serve {verifier} --dtype bfloat16 \\",
          f"  --speculative-config '{{\"model\": \"{repo_id}\", \"num_speculative_tokens\": {cfg.get('block_size', 8)}, \"method\": \"dspark\"}}'",
          "```", "",
          "`config.json` points to `config.py` (the custom `DSparkSpeculatorConfig`), which the `speculators` library provides when installed.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--repo-id", required=True, help="<namespace>/<name>, or just <name> for the logged-in account")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--description", default="", help="experiment-specific paragraph for the model card")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit-message", default="Upload drafter")
    args = ap.parse_args()

    root = Path(args.checkpoints_dir)
    best, latest = pick(root)
    repo_id = args.repo_id if "/" in args.repo_id else f"<logged-in account>/{args.repo_id}"
    main_ck = best or latest
    parts = [("", main_ck)] + ([("latest/", latest)] if best is not None and best != latest else [])
    print(f"checkpoints: {root}\nrepo: {repo_id} ({'public' if args.public else 'private'} if created)")
    print(f"best: {best.name if best else 'none'} | latest: {latest.name} (epoch, step) = {state_key(latest)}")
    for prefix, d in parts:
        for f in FILES:
            if (d / f).exists():
                print(f"  {prefix + f:36s} {(d / f).stat().st_size / 1e6:9.1f} MB   (from {d.name}/)")
    text = card(root, best, latest, repo_id, args.run_name, args.description)
    print("--- card ---\n" + text + "--- end card ---")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"no usable Hugging Face login ({type(e).__name__}); set HF_TOKEN or run `hf auth login`")
    print(f"logged in as {who}")
    if "/" not in args.repo_id:
        args.repo_id = f"{who}/{args.repo_id}"
        text = card(root, best, latest, args.repo_id, args.run_name, args.description)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for prefix, d in parts:
            (Path(tmp) / prefix).mkdir(parents=True, exist_ok=True)
            for f in FILES:
                if (d / f).exists():
                    (Path(tmp) / prefix / f).symlink_to((d / f).resolve())
        (Path(tmp) / "README.md").write_text(text)
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id, repo_type="model", commit_message=args.commit_message)
    info = api.model_info(args.repo_id)
    print(f"uploaded: https://huggingface.co/{args.repo_id} (private={info.private}, {len(info.siblings or [])} files)")


if __name__ == "__main__":
    main()
