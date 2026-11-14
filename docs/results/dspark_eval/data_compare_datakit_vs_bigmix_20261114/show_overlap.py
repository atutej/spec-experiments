import numpy as np, re, sys, pyarrow as pa
from transformers import AutoTokenizer
W="/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"
tok=AutoTokenizer.from_pretrained("open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom",trust_remote_code=True)
z=np.load(f"{W}/results/contam_index.npz",allow_pickle=True); setof=z["set_of"]; texts=z["text"]
def row_of(shard, r):
    for i,b in enumerate(pa.ipc.open_stream(pa.memory_map(shard))):
        if r < 1000*(i+1):
            return np.array(b.column(0)[r-1000*i].as_py()), np.array(b.column(1)[r-1000*i].as_py())
        # (batches are 1000 rows)
want=[("bigmix_tg","HumanEval"),("bigmix_tg","MATH-500"),("datakit_orig","MT-Bench"),("datakit_orig","MATH-500"),("datakit_orig","GSM8K"),("bigmix_tg","GSM8K")]
for ds,s in want:
    c=np.load(f"{W}/results/contam_{ds}.npz",allow_pickle=True); m=np.where(setof==s)[0]; b=c["best"][m]; k=m[int(np.argmax(b))]
    si,r=c["ex"][k]; ids,mask=row_of(str(c["shards"][si]),int(r)); prompt=tok.decode(ids[mask==0][:400].tolist(),skip_special_tokens=False)
    q=re.sub(r"\s+"," ",texts[k])[:110]
    i=prompt.find(q[:40]); snippet=re.sub(r"\s+"," ",prompt[max(0,i):max(0,i)+160] if i>=0 else prompt[200:360])
    print(f"== {ds} / {s}: best coverage {b.max():.2f}\n   eval problem : {q}\n   training row : {snippet}\n")
