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
      [--description TEXT] [--public] [--dry-run] [--datasets ID ...] [--speculators-ref BRANCH] \
      [--data-manifest manifest.json] [--card-extra FILE.md] [--serve-args "..."]

`--data-manifest` is the dataset's manifest.json (per-source row and token counts, written by build_datakit_sft_mix.py write). It is uploaded as
`training_data_manifest.json` and the card gets a "Training data" section that summarises it and links it, as the Grug target's card does for
`training_data_mix.json`. `--card-extra` appends a markdown file (e.g. results tables) before the "Use" section.

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
            key = ".".join(stack[: indent + 1]) if stack else ""
            out.setdefault(key, []).append(line[2:])
            continue
        k, _, v = line.partition(":")
        stack = stack[:indent] + [k]
        if v.strip():
            out[".".join(stack)] = v.strip()
    return out


def data_section(manifest: dict, name: str) -> list[str]:
    t = manifest.get("totals", {})
    src = manifest.get("per_source", {})
    L = ["## Training data", "",
         f"Prompts and (for itw drafters) replies come from the Datakit SFT mixture the target was fine-tuned on: {len(src)} SFT leaf sources, "
         f"{t.get('rows_kept', 0):,} rows in {manifest.get('shards', '?')} shards, {t.get('tokens_kept', 0) / 1e9:.2f}B tokens after clipping rows at "
         f"{manifest.get('seq_length', '?'):,} tokens ({t.get('rows_clipped', 0):,} rows clipped, {t.get('tokens_clipped', 0) / 1e9:.2f}B tokens removed), of which "
         f"{t.get('tokens_supervised', 0) / 1e9:.2f}B are supervised assistant tokens.", "",
         f"[`{name}`]({name}) is that dataset's manifest: totals, the shuffle seed, and per-source row and token counts (`rows_in`, `rows_kept`, "
         "`tokens_kept`, `tokens_supervised`, `rows_clipped`, `tokens_clipped`, `dropped_unsupervised`). The source allocations themselves are in Marin's "
         "[`special_token_1000_mix.json`](https://github.com/marin-community/marin/blob/82620ad62bbe23b9b12a89d656c24e6f80aff91a/experiments/grug_sft/special_token_1000_mix.json) "
         "and the target repo's [`training_data_mix.json`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/blob/main/training_data_mix.json). "
         "The last 0.1% of rows is held out for validation.", ""]
    return L


def card(root: Path, best: Path | None, latest: Path, repo_id: str, run_name: str, description: str, datasets: list[str],
         speculators_ref: str, manifest: dict | None, manifest_name: str, card_extra: str, serve_args: str, wandb_runs: list[str], reproduce: str = "") -> str:
    y = yaml_flat(root / "run.yaml")
    shown = best or latest
    cfg = json.loads((shown / "config.json").read_text())
    layer_cfg = cfg.get("transformer_layer_config", {})
    layers = layer_cfg.get("num_hidden_layers", y.get("train.draft.num_layers", "?"))
    taps = cfg.get("aux_hidden_state_layer_ids") or y.get("train.draft.target_layer_ids", "?")
    verifier = cfg.get("speculators_config", {}).get("verifier", {}).get("name_or_path") or y.get("train.verifier.verifier_name_or_path", "?")
    val = json.loads((shown / "val_metrics.json").read_text()) if (shown / "val_metrics.json").exists() else None
    hub_id = verifier if "/" in verifier and not verifier.startswith("/") else None
    fm = ["---"] + ([f"base_model: {hub_id}"] if hub_id else []) + (["datasets:"] + [f"- {d}" for d in datasets] if datasets else []) + [
        "tags:", "- speculative-decoding", "- dspark", "- drafter", "---", ""]
    L = fm + [f"# {repo_id.split('/')[-1]}", "",
              f"A DSpark speculative-decoding drafter (block size {cfg.get('block_size')}, {layers} layers, reduced "
              f"draft vocabulary {cfg.get('draft_vocab_size')}) for the target model **{verifier}**. Trained with the `speculators` library "
              f"(fork `atutej/speculators`, branch `{speculators_ref}`); run name `" + run_name + "`.", ""]
    if description:
        L += [description, ""]
    if manifest is not None:
        L += data_section(manifest, manifest_name)
    L += ["## Training", "",
          f"- Online training: hidden states of layers {taps} (plus the final layer) of the target, streamed by vLLM through Mooncake",
          f"- Sequence length {y.get('train.data.total_seq_len', '?')}, max anchors {y.get('train.data.max_anchors', '?')}, learning rate "
          f"{y.get('train.optimizer.lr', '?')}, {y.get('train.trainer.epochs', '?')} epoch(s), loss {y.get('train.loss.loss_fn', '?')}",
          f"- Markov head rank {y.get('train.dspark.markov_rank', '?')} ({y.get('train.dspark.markov_head_type', '?')}), confidence head "
          f"{y.get('train.dspark.enable_confidence_head', '?')}"]
    if wandb_runs:
        L += ["", "## Training logs", "", "Weights & Biases (project `atutej/marin_speculator`):", ""]
        for item in wandb_runs:
            label, _, url = item.partition("=")
            L.append(f"- [{label}]({url})")
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
    if card_extra:
        L += ["", card_extra.strip(), ""]
    if reproduce:
        L += ["", reproduce.strip(), ""]
    L += ["", "## Use", "",
          "With the Marin vLLM fork (supports `method: dspark`):", "",
          "```", f"vllm serve {verifier} {serve_args} \\",
          f"  --speculative-config '{{\"model\": \"{repo_id}\", \"num_speculative_tokens\": {cfg.get('block_size', 8)}, \"method\": \"dspark\", "
          "\"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"standard\"}'",
          "```", "",
          "Once it is up (the server listens on port 8000), query it as usual; speculative decoding is transparent to the client:", "",
          "```", "curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{",
          f"  \"model\": \"{verifier}\",",
          "  \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python function that checks whether a string is a palindrome, and explain it briefly.\"}],",
          "  \"temperature\": 1.0, \"max_tokens\": 600}'", "```", "",
          "The chat template defaults to `Reasoning: /think`, so the reply starts with the model's reasoning as plain text inside `content` (no reasoning parser is enabled in this command).", "",
          "The two sampling fields are not stored in the checkpoint: vLLM defaults to greedy drafting, which measured lower acceptance for sampled decoding (T=1). "
          "vLLM reads this checkpoint's `speculators_config` itself; `config.json` also points to `config.py` (the custom `DSparkSpeculatorConfig`), "
          "which the `speculators` library provides when installed.", ""]
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
    ap.add_argument("--datasets", nargs="*", default=["nvidia/Nemotron-Terminal-Corpus"], help="dataset ids for the card's front matter (default: the Qwen3-0.6B runs' corpus; pass none for no list)")
    ap.add_argument("--speculators-ref", default="nemotron-terminal-preset", help="branch of the speculators fork named in the card")
    ap.add_argument("--data-manifest", default="", help="the training dataset's manifest.json: uploaded as training_data_manifest.json and summarised in the card")
    ap.add_argument("--card-extra", default="", help="markdown file appended to the card before the Use section (e.g. results tables)")
    ap.add_argument("--wandb-run", action="append", default=[], metavar="LABEL=URL", help="a W&B run to list in the card's Training logs section (repeatable)")
    ap.add_argument("--reproduce", default="", help="markdown file for a 'Reproduce' section; {GH_REF} in it is replaced by --gh-ref")
    ap.add_argument("--gh-ref", default="", help="commit SHA of the code repo the Reproduce links point at (required to upload when --reproduce is used)")
    ap.add_argument("--serve-args", default="--dtype bfloat16", help="extra `vllm serve` flags shown in the card's Use section")
    args = ap.parse_args()

    root = Path(args.checkpoints_dir)
    best, latest = pick(root)
    manifest_name = "training_data_manifest.json"
    manifest = json.loads(Path(args.data_manifest).read_text()) if args.data_manifest else None
    card_extra = Path(args.card_extra).read_text() if args.card_extra else ""
    MARK = "<GH-COMMIT-AFTER-PUSH>"
    reproduce = Path(args.reproduce).read_text().replace("{GH_REF}", args.gh_ref or MARK) if args.reproduce else ""
    if MARK in reproduce and not args.dry_run:
        sys.exit("the Reproduce links need --gh-ref <pushed commit SHA>; refusing to upload a card with placeholder links")
    datasets = [d for d in args.datasets if d]
    repo_id = args.repo_id if "/" in args.repo_id else f"<logged-in account>/{args.repo_id}"
    main_ck = best or latest
    parts = [("", main_ck)] + ([("latest/", latest)] if best is not None and best != latest else [])
    print(f"checkpoints: {root}\nrepo: {repo_id} ({'public' if args.public else 'private'} if created)")
    print(f"best: {best.name if best else 'none'} | latest: {latest.name} (epoch, step) = {state_key(latest)}")
    for prefix, d in parts:
        for f in FILES:
            if (d / f).exists():
                print(f"  {prefix + f:36s} {(d / f).stat().st_size / 1e6:9.1f} MB   (from {d.name}/)")
    if manifest is not None:
        print(f"  {manifest_name:36s} {Path(args.data_manifest).stat().st_size / 1e6:9.1f} MB   (from {args.data_manifest})")
    text = card(root, best, latest, repo_id, args.run_name, args.description, datasets, args.speculators_ref, manifest, manifest_name, card_extra, args.serve_args, args.wandb_run, reproduce)
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
        text = card(root, best, latest, args.repo_id, args.run_name, args.description, datasets, args.speculators_ref, manifest, manifest_name, card_extra, args.serve_args, args.wandb_run, reproduce)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for prefix, d in parts:
            (Path(tmp) / prefix).mkdir(parents=True, exist_ok=True)
            for f in FILES:
                if (d / f).exists():
                    (Path(tmp) / prefix / f).symlink_to((d / f).resolve())
        if manifest is not None:
            (Path(tmp) / manifest_name).symlink_to(Path(args.data_manifest).resolve())
        (Path(tmp) / "README.md").write_text(text)
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id, repo_type="model", commit_message=args.commit_message)
    info = api.model_info(args.repo_id)
    print(f"uploaded: https://huggingface.co/{args.repo_id} (private={info.private}, {len(info.siblings or [])} files)")


if __name__ == "__main__":
    main()
