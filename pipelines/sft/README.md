# SFT pipeline (planned, not built yet)

Fine-tune a base model on a Hugging Face dataset. First goal: SFT Qwen3-0.6B on the Nemotron-Terminal 100k sample, then rerun the
off-policy drafter experiment (`pipelines/speculator_training`) with that checkpoint as the target model (a new experiment folder
there, with the target path as a setting). SFT is kept apart from speculator training in the workspace: its own pipeline, its own
conda env.

## Decisions so far (discussion, nothing implemented)

- **Trainer: LLaMA-Factory, upstream `hiyouga/LlamaFactory`**, installed from source as a sibling repo of `speculators/` and
  `vllm/` and pinned by SHA in `setup/repos.txt` (candidate: `ce9dc9e072f8`, main on 2026-09-28, 72 commits after v0.9.5). We reuse
  OpenThoughts-Agent's YAML configs as a starting point (e.g. `sft/lf_configs/qwen3/`), NOT its `hpc.launch` launcher.
  Why not OpenThoughts-Agent's fork (`mlfoundations/LLaMA-Factory`, pinned `d20b86665a27`, 2026-07-02): its history is unrelated to
  upstream's and the trees differ in 576 files. Its own ~256 commits are mostly infra (transformers v5 fixes, ALST/Ulysses sequence
  parallelism, chunked cross-entropy for big vocabularies, Qwen3.5 extras) but also OpenThoughts-specific code (delphi template,
  database upload), and it LACKS upstream's `discarding_history_cot` / `preserve_thinking`, which with `mask_history` strip earlier
  turns' thinking without inserting empty think tags (what we need). Its multi-turn Qwen3 path keeps history thinking and inserts empty
  think tags. If LLaMA-Factory is ever needed for a large model, port ALST / chunked loss onto upstream instead.
  `marin-community` has no LLaMA-Factory fork. Not yet run: confirm tokenization and masks against `apply_chat_template`.
- **Data: exactly the 100k-conversation sample the speculator pipeline uses** (seed 0), made by the same export code
  (`tools/export_registry_dataset.py`) and checked by hash against the existing sample file, so SFT and drafter training see the
  same conversations and the rollout-eval prompts (which exclude that sample) stay held out.
- **Loss:** one training example per assistant turn; loss only on that turn's thinking and response; earlier turns, user and tool
  turns are context, and earlier assistant turns have their thinking stripped (what the Qwen3 chat template does at inference).
  Verify LLaMA-Factory's tokenization and masks against HF `apply_chat_template` on sample rows before any real run.
- **Scope:** off-policy rerun only for now. **W&B project/naming:** to be decided after the rest is done.

## Possible later: a Levanter backend (likely needed for Snowball)

Now: Qwen3-0.6B. Later: Marin's Snowball (67B-A2B MoE, `model_type=grug_moe`). OpenThoughts-Agent's marin notes say SFT of that
model stays on the native Levanter path (`run_grug`) and that Levanter's Snowball module is load/score only, so LLaMA-Factory is
probably not an option for it (not verified). Plan for a Levanter backend then.

We may add a second SFT backend on Levanter (JAX, in `marin-community/marin` under `lib/levanter`; the standalone Levanter repo
is stale since 2026-01) if LLaMA-Factory is not enough or we want the marin-native trainer. What we know from reading its repo
(nothing was run): current `pyproject.toml` pins `jax[cuda13]==0.11.1` with B200 and aarch64 notes; it has `Qwen3LMHeadModel`,
HF conversion in both directions (`export_lm_to_hf.py`) and chat SFT (`main/sft.py`). Open problems: a new JAX env on Vista
(aarch64, CUDA 13, GB200), and loss masking, because its chat masks need a `{% generation %}` block that Qwen3's template lacks
and it masks every assistant turn, while we want last-turn-only masks (a custom template, or pre-tokenized input with our own
mask, which is unconfirmed). If we try it: a time-boxed spike on a gb idev node (env, `jax.devices()`, a tiny Qwen3-0.6B SFT).
