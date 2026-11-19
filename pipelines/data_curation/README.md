# data_curation

Reproducible builds of the SFT datasets the drafters train on. A mix is built from public Hugging Face sources through Marin's own
datakit code (download -> chat convert -> render -> normalize -> tokenize), at the Marin commit that built the original, and written as Arrow
shards of `input_ids` / `loss_mask` / `seq_len` (the layout `speculators` trains on) plus `token_freq.{json,pt}` and `manifest.json`.

```bash
export HF_TOKEN=<token>                          # some sources are private
bash tools/prepare.sh <TARGET_DIR> datakit_sft_1005          # any machine; everything lands under TARGET_DIR; rerun to resume
PARTITION=gg ACCOUNT=CCR24067 TIME=2-00:00:00 bash tools/slurm/submit.sh <TARGET_DIR> datakit_sft_1005   # same, on one exclusive node (login node only)
```

Stages (`STAGES="env plan fetch stats select write finalize"`, each skips finished work): `env` clones Marin at the pinned commit and builds a
uv venv (`CC=gcc`, since TACC aarch64 puts NVIDIA `nvc` first on PATH and it cannot build `pycosat`); `plan` resolves every source to pinned HF
repos and sizes them; `fetch` runs Marin's download/normalize steps; `stats` tokenizes and counts every row (and checks the loss mask against the
template); `select` picks the rows; `write` tokenizes again, clips at `SEQ_LENGTH` (32768), shuffles and writes the shards; `finalize` writes
`token_freq.pt` and runs the trainer-style load check (set `SPECULATORS_PY` to a python with torch + datasets; the Marin venv has no torch).
Knobs: `SOURCES_FILE` / `ROWS` (a small test run), `CONCURRENT`, `WORKERS`, `SEQ_LENGTH`.

## Mixes (`mixes/pins.json`)

| mix | release | Marin commit | sources | allocation | selection |
|---|---|---|---|---|---|
| `datakit_sft_0921` | `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` | `82620ad` | 203 | 53.7B tokens | seeded random rows up to each source's allocation |
| `datakit_sft_1005` | `...-2026.10.05` @ `ac60d940` | `ce7c77a` | 216 | 253.9B packed positions (~234B tokens) | every row of every source, once |

`datakit_sft_1005` is what Snowball and the Mochi SFT trained on (supervised part only: the release also has ~20% pretraining replay from Marin's
private Harrier stores, not reproduced here; the replay spec is vendored as `mixes/datakit_sft_1005_replay_skew8.json`). 183 sources are shared with
0921; 33 are new, 20 dropped, weights differ (finance 17.6% -> 3.9%). The mix specs are the files committed in Marin at the pinned commits.
`experiments/<mix>/NOTES.md` records sizes and verification results.
