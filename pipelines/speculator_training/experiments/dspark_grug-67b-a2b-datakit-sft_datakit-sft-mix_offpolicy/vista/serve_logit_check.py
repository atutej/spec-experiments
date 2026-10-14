"""Validate the tapped final-layer hidden state and Grug's gated head against the live server's own log-probabilities.

For a prompt, the server returns log P(token_i | tokens_<i) (`prompt_logprobs`). The same numbers are recomputed here from the
extracted layer-26 hidden state with the checkpoint's weights: lm_head(norm(h)) with and without the gated final norm
(h * sigmoid(up(silu(down(h))))). The variant that matches shows what the tap is and what the DSpark TV-loss targets must use.
"""
import argparse, glob, json, os, time
import openai, torch
from safetensors import safe_open
from transformers import AutoTokenizer
import serve_smoke_client_util as u  # wait_for_file, shared with serve_smoke_client.py

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000); ap.add_argument("--model", required=True)
ap.add_argument("--layers", type=int, nargs="+", required=True); ap.add_argument("--hs-dir", required=True)
ap.add_argument("--n", type=int, default=700); ap.add_argument("--device", default="cuda:2")
a = ap.parse_args()
client = openai.Client(base_url=f"http://localhost:{a.port}/v1", api_key="x", timeout=1800)
tok = AutoTokenizer.from_pretrained(a.model)
text = open("/scratch/09749/atutej/marin_speculator/speculators/README.md").read() * 20
ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"][: a.n - 1]

# server side: hidden states, and the model's own prompt log-probs
res = client.completions.create(model=a.model, prompt=ids, max_tokens=1, extra_body={"return_token_ids": True})
path = res.kv_transfer_params.get("hidden_states_path") or res.kv_transfer_params.get("handle"); u.wait_for_file(path)
from safetensors.torch import load_file
h = load_file(path)["hidden_states"]                                   # [n, taps, 2560] bf16
r2 = client.completions.create(model=a.model, prompt=ids, max_tokens=1, extra_body={"prompt_logprobs": 0})
plp = r2.choices[0].model_extra["prompt_logprobs"]                     # [None, {tok_id: {logprob,...}}, ...]
server = torch.tensor([plp[i][str(ids[i])]["logprob"] for i in range(1, len(ids))])

# checkpoint weights (the shards are in the local HF cache)
snap = glob.glob(os.path.expanduser(os.environ["HF_HOME"] + "/hub/models--*Grug*/snapshots/*"))[0]
wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def W(name):
    with safe_open(f"{snap}/{wm[name]}", framework="pt", device="cpu") as f: return f.get_tensor(name)
dev = a.device
norm_w, down, up, head = (W(n).to(dev).float() for n in ("model.norm.weight", "model.final_gated_norm.down_proj.weight",
                                                         "model.final_gated_norm.up_proj.weight", "lm_head.weight"))
cfg = json.load(open(f"{snap}/config.json")); eps = cfg["rms_norm_eps"]
last = h[:, -1].to(dev).float()                                        # tap order: the requested layers, then the final layer
print(f"{len(ids)} tokens; tap shapes {tuple(h.shape)}; layer-26 tap RMS per token: mean {last.pow(2).mean(-1).sqrt().mean():.2f}; norm weight mean {norm_w.mean():.2f}")

def lp(hidden):                                                        # log P(token_{i+1}) from the hidden state at position i
    logits = hidden @ head.T
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(ids[1:], device=dev)[:, None])[:, 0].cpu()[: len(ids) - 1]
def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
gate = lambda x: x * torch.sigmoid(torch.nn.functional.silu(x @ down.T) @ up.T)
rows = {"tap -> lm_head (no norm)": last[:-1], "norm -> lm_head": rms(last, norm_w)[:-1], "norm -> GATE -> lm_head": gate(rms(last, norm_w))[:-1],
        "norm(1+w) -> GATE -> lm_head": gate(rms(last, 1 + norm_w))[:-1], "GATE -> norm -> lm_head": rms(gate(last), norm_w)[:-1]}
print(f"{'variant':32s} {'mean |diff| (nats)':>20s} {'max |diff|':>12s} {'corr':>8s}   (server prompt log-probs vs recomputed)")
with torch.no_grad():
    for name, hid in rows.items():
        mine = lp(hid)[: len(server)]
        d = (mine - server).abs()
        c = torch.corrcoef(torch.stack([mine, server]))[0, 1]
        print(f"{name:32s} {d.mean():20.4f} {d.max():12.3f} {c:8.4f}")
print("server mean log-prob %.3f" % server.mean())
