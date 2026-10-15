# Serving smoke test: DSpark drafter on Grug-67B-A2B-Datakit-SFT-262K (no RL)

Stage 1 of `rl_speculation` (README). Status: run once on 2026-10-14 (results below). One run, one seed per request, no repeats.

## Question
Does vLLM (the `vllm` env, pin 39e62869, torch 2.13/cu132) start the Datakit-SFT target with our DSpark drafter under DP x EP on one GB200
node, and what accept length and tok/s does it give on R2E-Gym-style prompts? Plain decoding is the control.

## What the two arms share
Same server flags (TP=1, DP=4, expert-parallel, max-model-len 32768), same prompts, T=1 / top-p 1 / top-k off, per-request seeds, 16 concurrent
streams. The only difference is `--speculative-config {method: dspark, num_speculative_tokens: 8, model: <drafter>}`.

## Read these, in this order
1. `dspark/server.log` reaches "ready". A startup assertion "All drafting layers should belong to the same kv cache group" means the EAGLE-3-only
   `_co_locate_eagle3_layers` gate needs `dspark` (see docs/rl_dspark_infra.md).
2. `dspark/acceptance.csv`: accept length (1 + accepted/drafts) and per-position acceptance, from vLLM's own counters.
3. `plain/results.json` vs `dspark/results.json`: tok/s per stream and per node. A dspark step slower than plain despite accept length > 2 points at
   the DP-sync gate (`draft_requires_dp_sync` is True for dspark). Re-run with `VLLM_DRAFT_DP_SYNC` if the pin has it, or patch the gate.

## Limits of this test (read before quoting a number)
- Prompts are R2E-Gym instruction first turns, about 900 tokens each. RL contexts reach 49k and have many turns; Luke measured 2.15 tokens/step for
  prompts under 16k and 2.55 above 40k on his 64k Snowball (Jupiter GH200, EAGLE-3, adapted draft; issue #9079).
- Not a measured result from our setup until run. Luke's numbers are the reference, not a target we have met.
- The drafter was trained on offline Datakit text, and its validation accept length (2.926, teacher-forced) is not comparable to rollout accept length.
- No R2E-Gym adaptation and no agent system prompt.

## Run
```
sbatch -p gb vista/run.sbatch prompts plain dspark      # from a login node; or on a 4-GPU idev: bash vista/run.sh prompts plain dspark
MAX_PROMPTS=8 MAX_TOKENS=512 bash vista/run.sh dspark   # quick check
```
Outputs: `runs/rl_speculation/serving_smoke/{plain,dspark}/`.

## Results (achieved, 2026-10-14; Vista GB200 node, 4 GPUs, TP=1 x DP=4 x EP, vLLM 39e62869, 128 R2E-Gym first turns, T=1, 16 streams, max 4096 new tokens)
| arm | accept length | output tok/s (node) | wall | completions |
|---|---|---|---|---|
| plain | 1.00 | 1,696 | 84.0 s | 105 stop / 23 length, 1,113 tokens mean |
| dspark (8 spec tokens, unadapted drafter) | 2.053 over 67,131 drafts | 2,400 | 57.4 s | 105 stop / 23 length, 1,077 tokens mean |

- Per-position acceptance (dspark): 0.50, 0.25, 0.14, 0.08, 0.04, 0.03, 0.01, 0.006.
- Node throughput ratio 2,400 / 1,696 = 1.41x. It is total tokens over wall time, so it includes the tail of the longest requests, and the two arms
  sampled different texts (same per-request seeds, but speculative sampling does not reproduce plain sampling), hence the different token totals.
  Not a per-stream latency measurement.
- The server started with the 3-layer draft under DP x EP; no KV-group assertion, so `_co_locate_eagle3_layers` was not needed here (ready after 397 s).
  The DP-sync gate was not isolated: the speedup is measured with the gate as shipped (dense dspark draft does the per-pass all-reduce).
- Not a parity test: 0 of 128 completions are identical between arms at T=1, as expected. Output equivalence needs a greedy check or the RL stage's TIS ratio.
- Reference, not target: Luke's adapted EAGLE-3 draft gave 2.53 to 2.56 tokens per step and 1.61x per stream (GH200, 64k Snowball, issue #9114). Ours is unadapted, on
  short first-turn prompts (about 930 tokens), so the comparison is loose.
