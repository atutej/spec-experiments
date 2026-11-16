# Drafter A vs B on Dr. Doom: acceptance length per policy task

A = datakit-sft-mix_itw-20261108 (original Datakit-SFT text; 63.6% /think and 36.4% /nothink rows).
B = bigmix_itw-tg-20261113 (the target's own replies to BigMix prompts; 100% /think rows).

Two runs per drafter and task (the policy seed and seed 43), one server per drafter-task job, 4 GPUs, DP4 x EP4, 8 speculative tokens, concurrency as the policy sets it (16; TriviaQA 128). AL = 1 + accepted draft tokens / drafts. Per-run values: results.csv.

| Task | Thinking | A mean | B mean | A - B | run spread A / B | runs A / B |
|---|---|---|---|---|---|---|
| gsm8k-0shot | off | 4.166 | 4.114 | +0.051 | 0.002 / 0.022 | 2 / 2 |
| humanevalplus | off | 5.382 | 4.963 | +0.419 | 0.112 / 0.302 | 2 / 2 |
| mbppplus | off | 3.603 | 3.425 | +0.178 | 0.085 / 0.159 | 2 / 2 |
| ifbench | off | 2.117 | 2.282 | -0.165 | 0.048 / 0.204 | 2 / 2 |
| cruxeval | off | 2.672 | 2.936 | -0.264 | 0.085 / 0.084 | 2 / 2 |
| triviaqa | off | 2.048 | 2.128 | -0.080 | 0.002 / 0.134 | 2 / 2 |
| math500 | on | 3.517 | 3.701 | -0.184 | 0.021 / 0.003 | 2 / 2 |
| aime24 | on | 4.282 | 4.344 | -0.062 | 0.012 / 0.013 | 2 / 2 |
| gpqa-diamond | on | 4.173 | 3.812 | +0.360 | 0.035 / 0.057 | 2 / 2 |

Mean gap over tasks with both drafters:
- thinking off: +0.023 over 6 tasks
- thinking on: +0.038 over 3 tasks

Reading it:
- Two runs per drafter: the "run spread" is the absolute difference between the two runs, a rough noise yardstick, not a confidence interval. The runs differ through sampling nondeterminism; whether the Evalchemy seed reaches the sampler is unconfirmed.
- Task-by-task the gaps change sign and are larger than the difference between the thinking-off and thinking-on averages, so these numbers do not show a mode effect: the gap tracks the task.
- Where a task's sampling default is greedy the two runs are close to identical and the spread says nothing about noise.
- Not run: FinanceBench, MRCR.
