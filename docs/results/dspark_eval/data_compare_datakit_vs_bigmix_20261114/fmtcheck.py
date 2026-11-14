"""Exact eval-format check on a spread sample (first 1000 rows of 12 evenly spaced shards per dataset):
 exact suffix 'Please reason step by step, and put your final answer within \\boxed{}.' at the end of the first user turn; any boxed instruction; a \\boxed{ in the reply text."""
import glob, sys, json
import numpy as np, pyarrow as pa, pyarrow.compute as pc
from transformers import AutoTokenizer
sys.path.insert(0, "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114/scripts")
from datacmp import DATASETS, TOK
name = sys.argv[1]; d = DATASETS[name][0]; shards = sorted(glob.glob(f"{d}/data-*.arrow")); pick = shards[::max(1, len(shards) // 12)][:12]
tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
SUF = "Please reason step by step, and put your final answer within \\boxed{}."
c = dict(rows=0, exact_suffix=0, boxed_instr=0, step=0, reply_boxed=0, math_like=0, math_like_reply_boxed=0, exact_suffix_reply_boxed=0)
for p in pick:
    b = next(iter(pa.ipc.open_stream(pa.memory_map(p)))); ic, mc = b.column(0), b.column(1)
    lens = pc.list_value_length(ic).to_numpy(zero_copy_only=False).astype(np.int64); offs = np.concatenate([[0], np.cumsum(lens)])
    flat = pc.list_flatten(ic).to_numpy(zero_copy_only=False).astype(np.int64); mf = pc.list_flatten(mc).to_numpy(zero_copy_only=False) != 0
    for r in range(len(lens)):
        ids = flat[offs[r]:offs[r + 1]]; m = mf[offs[r]:offs[r + 1]]
        if not m.any(): continue
        fs = int(np.argmax(m)); tail = tok.decode(ids[max(0, fs - 90):fs].tolist(), skip_special_tokens=False); head = tok.decode(ids[:min(fs, 500)].tolist(), skip_special_tokens=False)
        rep = tok.decode(ids[m][:4000].tolist(), skip_special_tokens=False)
        ex = SUF in tail; bi = "boxed" in tail; rb = "\\boxed{" in rep; ml = bi or ("$" in head and len(ids[:fs]) < 600)
        c["rows"] += 1; c["exact_suffix"] += ex; c["boxed_instr"] += bi; c["step"] += ("step by step" in tail.lower()); c["reply_boxed"] += rb
        c["math_like"] += ml; c["math_like_reply_boxed"] += (ml and rb); c["exact_suffix_reply_boxed"] += (ex and rb)
print(json.dumps({name: c}))
