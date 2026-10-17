"""Client half of serve_smoke.sh: generation sanity, hidden-state shapes/statistics, prefill throughput."""
import argparse, fcntl, glob, json, os, time, concurrent.futures as cf
import openai, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000); ap.add_argument("--model", required=True)
ap.add_argument("--layers", type=int, nargs="+", required=True); ap.add_argument("--hs-dir", required=True)
a = ap.parse_args()
client = openai.Client(base_url=f"http://localhost:{a.port}/v1", api_key="x", timeout=1800)
tok = AutoTokenizer.from_pretrained(a.model)

print("== 1. generation (thinking on)")
r = client.chat.completions.create(model=a.model, max_tokens=400, temperature=0.6,
    messages=[{"role": "user", "content": "What is 17 times 23? Answer briefly."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}})
m = r.choices[0].message
print("reasoning:", (getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None) or "")[:300].replace("\n", " "))
print("content  :", (m.content or "")[:300].replace("\n", " "), "| finish:", r.choices[0].finish_reason)
r2 = client.chat.completions.create(model=a.model, max_tokens=60, temperature=0,
    messages=[{"role": "user", "content": "Name the capital of France."}], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print("nothink  :", (r2.choices[0].message.content or "")[:120].replace("\n", " "))

print("== 2. hidden states")
text = open(glob.glob("/scratch/09749/atutej/marin_speculator/speculators/README.md")[0]).read() * 40
ids_all = tok(text, add_special_tokens=False)["input_ids"]
def wait_for_file(path, timeout=120.0):
    """The connector writes the file asynchronously and holds an flock on <path>.lock while writing (as speculators' client does)."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline: raise TimeoutError(f"{path} never appeared")
        time.sleep(0.05)
    lock = str(path) + ".lock"
    if os.path.exists(lock):
        fd = os.open(lock, os.O_RDWR)
        while True:
            try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() > deadline: raise TimeoutError(f"{lock} stayed locked")
                time.sleep(0.05)
        os.close(fd)

def hs(n):
    ids = [tok.bos_token_id] + ids_all[: n - 1]
    t = time.time()
    res = client.completions.create(model=a.model, prompt=ids, max_tokens=1, extra_body={"return_token_ids": True})
    dt = time.time() - t
    p = res.kv_transfer_params.get("hidden_states_path") or res.kv_transfer_params.get("handle")
    wait_for_file(p)   # the timed part is the request; the file write is the connector's
    return p, len(ids), dt
path, n, dt = hs(512)
d = load_file(path) if str(path).endswith(".safetensors") else torch.load(path)
for k, v in d.items():
    print(f"  {k}: shape {tuple(v.shape)} dtype {v.dtype} finite {bool(torch.isfinite(v.float()).all())} mean|x| {v.float().abs().mean():.3f}")
print(f"  expected: {n} tokens, {len(a.layers) + 1} tapped layers (ids {a.layers} + final), hidden 2560")

print("== 3. prefill throughput (hidden states on)")
hs(2048)  # warm-up
for n in (4096, 16384, 32768):
    path, n_tok, dt = hs(n); print(f"  {n_tok:6d} tokens in {dt:6.2f} s = {n_tok/dt:9,.0f} tok/s")
with cf.ThreadPoolExecutor(4) as ex:
    t = time.time(); out = list(ex.map(lambda _: hs(16384), range(4))); dt = time.time() - t
print(f"  4 concurrent x 16384 tokens: {sum(o[1] for o in out)/dt:9,.0f} tok/s aggregate")
