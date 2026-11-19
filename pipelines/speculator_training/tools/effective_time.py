#!/usr/bin/env python3
"""Effective-time cost model for a drafter trained on original (free) + targetgen (paid) completions.

Original completions cost nothing; a targetgen row costs the time its replies took to generate (2-node wall clock, the same hardware as training).
A one-epoch run is launched and stopped by hand when the EFFECTIVE time reaches the target:

    effective(step) = training wall time(step) + regen cost of the targetgen rows seen by `step`

Rows are visited in a uniformly random order (the trainer permutes all training rows every epoch), so by step s the run has seen
s * tokens_per_step tokens, of which a share tg_tokens / (orig_tokens + tg_tokens) are targetgen, i.e. that many tokens / tokens_per_tg_row rows.
Regeneration already spent on rows the run never reaches is not counted (it is a sunk cost, by the stated rule).

Predict:   effective_time.py [--step-time 1.14] [--target-hours 24]
Live:      effective_time.py --step 61000 --train-hours 19.2     (train-hours: summed wall time of the run's jobs, queue waits excluded)
Defaults are the measured numbers of the earlier runs (see NOTES.md of the layers6-taps5 experiments); override all of them from the command line.
"""
import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--orig-tokens", type=float, default=45.99e9, help="Datakit original, tokens kept (manifest.json)")
ap.add_argument("--tg-tokens", type=float, default=3.22e9, help="BigMix targetgen, all tokens")
ap.add_argument("--tg-rows", type=float, default=296_000)
ap.add_argument("--tokens-per-step", type=float, default=120.2e3, help="Datakit run: 45.99B tokens / 382,747 steps")
ap.add_argument("--step-time", type=float, default=1.14, help="seconds per step; 1.14 = Datakit original (24 h / 75,834 steps); BigMix-TG ran at 0.85-0.9, Datakit-TG at 1.1. MEASURE on the mix")
ap.add_argument("--regen-h-per-1k-rows", type=float, default=0.0595, help="BigMix-TG: 10.7 h / 180k rows and 6.9 h / 116k rows (2 nodes). Datakit-TG was 0.0429")
ap.add_argument("--target-hours", type=float, default=24.0)
ap.add_argument("--warmup-frac", type=float, default=0.01)
ap.add_argument("--step", type=float, help="live mode: the run's current step")
ap.add_argument("--train-hours", type=float, help="live mode: training wall hours so far")
a = ap.parse_args()

total_tokens = a.orig_tokens + a.tg_tokens
tg_share = a.tg_tokens / total_tokens
tokens_per_tg_row = a.tg_tokens / a.tg_rows
epoch_steps = total_tokens / a.tokens_per_step
tg_rows_per_step = a.tokens_per_step * tg_share / tokens_per_tg_row
regen_s_per_step = tg_rows_per_step * a.regen_h_per_1k_rows / 1000 * 3600

def lr_frac(s):  # linear warmup, then linear decay to zero over the epoch
    w = a.warmup_frac * epoch_steps
    return s / w if s < w else max(0.0, 1 - (s - w) / (epoch_steps - w))

def report(s, train_h, regen_h):
    tok = s * a.tokens_per_step
    rows = s * tg_rows_per_step
    print(f"  step {s:,.0f} ({s/epoch_steps:.1%} of the epoch), {tok/1e9:.2f}B tokens seen, targetgen rows seen ~{rows:,.0f} of {a.tg_rows:,.0f} ({rows/a.tg_rows:.1%}), learning rate {lr_frac(s):.2f} of peak (unannealed)")
    print(f"  training {train_h:.2f} h + regeneration credit {regen_h:.2f} h = effective {train_h + regen_h:.2f} h")

print(f"mix: {a.orig_tokens/1e9:.2f}B original + {a.tg_tokens/1e9:.2f}B targetgen tokens, targetgen share {tg_share:.2%} of tokens "
      f"({a.tg_rows/(a.tg_rows + a.orig_tokens/7.1e3):.1%} of rows if original rows average 7.1k tokens); epoch ~{epoch_steps:,.0f} steps")
print(f"per step: {a.step_time:.3f} s training + {regen_s_per_step:.3f} s regeneration credit ({tg_rows_per_step:.2f} targetgen rows/step at {a.regen_h_per_1k_rows} h per 1k rows) = {a.step_time + regen_s_per_step:.3f} s effective")
if a.step is not None:
    th = a.train_hours if a.train_hours is not None else a.step * a.step_time / 3600
    print("live:"); report(a.step, th, a.step * regen_s_per_step / 3600)
    eff = th + a.step * regen_s_per_step / 3600
    print(f"  {a.target_hours - eff:+.2f} h to the {a.target_hours:g} h effective target")
else:
    s = a.target_hours * 3600 / (a.step_time + regen_s_per_step)
    print(f"stop point for {a.target_hours:g} h effective:"); report(s, s * a.step_time / 3600, s * regen_s_per_step / 3600)
    print("sensitivity to the (unmeasured) step time:")
    print("  step time   stop step   tokens   targetgen rows seen   training h   regen credit h")
    for st in (0.85, 1.0, 1.14, 1.3):
        ss = a.target_hours * 3600 / (st + regen_s_per_step)
        print(f"  {st:>7.2f} s  {ss:>9,.0f}  {ss*a.tokens_per_step/1e9:>5.2f}B  {ss*tg_rows_per_step:>10,.0f} ({ss*tg_rows_per_step/a.tg_rows:>4.0%})   {ss*st/3600:>8.2f}   {ss*regen_s_per_step/3600:>8.2f}")
    print("regeneration cost sensitivity (step time as given):")
    for c in (0.0429, 0.0595, 0.08):
        rs = tg_rows_per_step * c / 1000 * 3600
        ss = a.target_hours * 3600 / (a.step_time + rs)
        print(f"  {c} h/1k rows: stop at step {ss:,.0f} ({ss*a.tokens_per_step/1e9:.2f}B tokens)")
