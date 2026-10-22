"""vLLM worker extension for online drafter training with Aurora (https://github.com/togethercomputer/aurora).

Passed to vLLM as worker_extension_cls (the same pattern MarinSkyRL uses): its methods become callable on every worker through collective_rpc and see the
worker's `self.model_runner`. It adds three things on top of the Marin vLLM fork's own online EAGLE-3 capture (vllm/v1/spec_decode/online_eagle.py), without
editing the fork:

  ac_begin(config)          start a capture interval (the fork's begin_online_eagle_capture)
  ac_seal_publish(spec)     seal the interval to node-local disk, write every captured window into the Mooncake store in the key layout Aurora's
                            trainer reads ({key}_hs, {key}_ids, {key}_lhs), and return the keys and shapes (what Aurora's /push_sample needs)
  ac_reload_draft(path)     load a draft checkpoint saved by Aurora (save_draft_model_for_serving) into the running drafter

Aurora tensors per sample (T = prompt + completion - 1 tokens): hidden_states (T, 3H) bf16 = the three tapped layers concatenated, input_ids (T,) int64,
last_hidden_states (T, H) bf16 = the target's final hidden states after the final norm (the input of the target's lm_head); the loss is on the completion.
"""
from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path

import torch

_STORE = None


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return socket.gethostbyname(socket.gethostname())


def _mooncake_store(cfg: dict):
    """One Mooncake client per worker process, set up the way hs_connectors and Aurora's MooncakeHiddenStateStore do."""
    global _STORE
    if _STORE is None:
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        result = store.setup(
            cfg.get("local_hostname") or _local_ip(),
            cfg.get("metadata_server", "P2PHANDSHAKE"),
            int(cfg.get("global_segment_size", 4 * 1024**3)),
            int(cfg.get("local_buffer_size", 2 * 1024**3)),
            cfg.get("protocol", "tcp"),
            cfg.get("device_name", ""),
            cfg["master_server_address"],
        )
        if result != 0:
            raise RuntimeError(f"Mooncake store setup failed with status {result}")
        _STORE = store
    return _STORE


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class AuroraCaptureExt:
    def ac_begin(self, config):
        """Begin a capture interval on this worker (requests must be admitted after this call)."""
        r = self.model_runner
        if r.speculative_config is not None and r.speculative_config.method == "eagle3":
            return r.begin_online_eagle_capture(dict(config))
        # DSpark runs on the fork's model runner V2, which already has the capture hooks (record_forward / record_sampled / finalize_request_length) and the same begin/seal
        # methods, but its begin only accepts method "eagle3". So do what its begin_online_eagle_capture does with that one check passed (its aux-layer lookup reads DSpark's layer ids).
        # A fork change would be: accept "dspark" in resolve_online_eagle_capture_config (README, deviation 4).
        if not hasattr(r, "_resolve_online_eagle_aux_layers"):
            raise RuntimeError(f"capture for method {r.speculative_config.method!r} needs the fork's model runner V2")
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.v1.spec_decode.online_eagle import OnlineEagleCapture, resolve_online_eagle_capture_config

        cfg = resolve_online_eagle_capture_config(
            dict(config), speculative_method="eagle3", async_scheduling=bool(r.scheduler_config.async_scheduling), pipeline_parallel_size=get_pp_group().world_size,
            worker_rank=r.dp_rank, max_window_tokens=r.max_model_len, aux_layer_ids=r._resolve_online_eagle_aux_layers)
        r.online_eagle_capture = OnlineEagleCapture(cfg)
        return {"active": True, "worker_rank": cfg.worker_rank, "step": cfg.step}

    def ac_seal_publish(self, spec):
        """spec: {"mooncake": {...store config...}, "keys": {request_id: mooncake key}, "scratch_dir": optional}. Returns one record per published window."""
        from safetensors.torch import load_file

        store = _mooncake_store(spec["mooncake"])
        keys = spec["keys"]
        tmp = tempfile.mkdtemp(prefix="aurora-capture-", dir=spec.get("scratch_dir") or None)
        try:
            out = os.path.join(tmp, "capture")
            manifest = self.model_runner.seal_online_eagle_capture(out)
            records = []
            for w in manifest["windows"]:
                rid = w["request_id"]
                # vLLM appends a random suffix to the external request id internally ("0" -> "0-8ddaffc8"); the caller keys samples by the external id
                if rid in keys:
                    key = keys[rid]
                elif rid.rsplit("-", 1)[0] in keys:
                    key = keys[rid.rsplit("-", 1)[0]]
                else:
                    continue
                t = load_file(os.path.join(out, w["path"]))
                hs, ids, lhs, mask = t["hidden_states"], t["input_ids"], t["head_input_hidden_states"], t["loss_mask"]
                prompt_tokens = int(mask.nonzero()[0]) if bool(mask.any()) else int(mask.numel())
                for suffix, tensor in (("_hs", hs), ("_ids", ids), ("_lhs", lhs)):
                    status = store.put(key + suffix, _tensor_bytes(tensor))
                    if status != 0:
                        raise RuntimeError(f"Mooncake put failed for {key + suffix} with status {status}")
                records.append({
                    "request_id": rid, "mooncake_key": key,
                    "tensor_shapes": {"hidden_states": list(hs.shape), "input_ids": list(ids.shape), "last_hidden_states": list(lhs.shape)},
                    "tensor_dtypes": {"hidden_states": str(hs.dtype), "input_ids": str(ids.dtype), "last_hidden_states": str(lhs.dtype)},
                    "prompt_tokens": prompt_tokens, "completion_tokens": int(w["supervised_tokens"]) + 1,
                    "window_tokens": int(w["tokens"]),
                })
            return {"windows": records, "captured_rows": manifest.get("captured_rows"), "dropped_requests": manifest.get("dropped_requests"),
                    "dropped_windows": manifest.get("dropped_windows"), "aux_layer_ids": manifest.get("aux_layer_ids")}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def ac_reload_draft(self, model_dir):
        """Load a draft checkpoint directory (pytorch_model.bin or *.safetensors with HF names, as Aurora saves it) into the running drafter."""
        draft = self.model_runner.get_draft_model()
        if draft is None:
            raise RuntimeError("no draft model is loaded on this worker")
        d = Path(model_dir)
        files = sorted(d.glob("*.safetensors"))
        if files:
            from safetensors.torch import load_file
            weights = {}
            for f in files:
                weights.update(load_file(str(f)))
        elif (d / "pytorch_model.bin").exists():
            weights = torch.load(d / "pytorch_model.bin", map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"no safetensors or pytorch_model.bin in {d}")
        # Aurora saves a full state dict, embed_tokens included. When the drafter shares the target's embedding table (vLLM's EAGLE-3 does unless the checkpoint it was
        # started from carried its own), loading it would overwrite the TARGET's embeddings: skip it then.
        skipped = []
        try:
            from vllm.v1.spec_decode.online_eagle import target_embedding_weight
            shared = draft.model.embed_tokens.weight.data_ptr() == target_embedding_weight(self.model_runner.get_model()).data_ptr()
        except Exception:  # noqa: BLE001
            shared = False
        if shared:
            skipped = [k for k in weights if "embed_tokens" in k]
            weights = {k: v for k, v in weights.items() if "embed_tokens" not in k}
        loaded = draft.load_weights(iter(weights.items()))
        torch.cuda.synchronize()
        return {"tensors_loaded": len(weights), "skipped_shared_embedding": skipped, "loaded": len(loaded) if loaded is not None else None}

    def ac_draft_checksum(self):
        """Sum of |w| over the drafter's floating-point parameters (float64): proves a reload changed the weights."""
        draft = self.model_runner.get_draft_model()
        total, n = 0.0, 0
        for prm in draft.parameters():
            if prm.is_floating_point():
                total += float(prm.detach().abs().double().sum()); n += prm.numel()
        return {"abs_sum": total, "numel": n}
