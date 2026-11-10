"""A fake OpenAI-compatible server for a CPU dry run of the AIPerf client path (no model). Streams a few chunks per chat request and keeps vLLM-style
speculative-decoding counters on /metrics, so we can check the flags, the multi-turn dispatch, the request count and the acceptance report before using GPU time.
Run: python fake_server.py PORT MODEL_NAME"""
import json, random, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
LOCK = threading.Lock(); C = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "requests": 0, "turns_with_history": 0}
rng = random.Random(0)
SEEN = []


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/health"): self._send(200, "ok", "text/plain")
        elif self.path.startswith("/v1/models"): self._send(200, json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]}))
        elif self.path.startswith("/stats"):
            import collections
            with LOCK:
                firsts = collections.Counter(h for h, k in SEEN if k == 1)
                depth = collections.Counter(k for h, k in SEEN)
            self._send(200, json.dumps({**C, "unique_first_prompts": len(firsts), "first_turn_requests": sum(firsts.values()), "repeated_first_prompts": sum(1 for v in firsts.values() if v > 1), "by_depth": dict(depth)}))
        elif self.path.startswith("/metrics"):
            with LOCK:
                t = "".join(f"# HELP vllm:{n}_total fake\n# TYPE vllm:{n}_total counter\nvllm:{n}_total{{engine=\"0\",model_name=\"{MODEL}\"}} {v}\n" for n, v in
                            (("spec_decode_num_drafts", C["drafts"]), ("spec_decode_num_draft_tokens", C["draft_tokens"]), ("spec_decode_num_accepted_tokens", C["accepted"])))
            self._send(200, t, "text/plain")
        else: self._send(404, "{}")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = body.get("messages", [])
        with LOCK:
            C["requests"] += 1; C["turns_with_history"] += int(len(msgs) > 1); C["max_msgs"] = max(C.get("max_msgs", 0), len(msgs)); C.setdefault("asst_in_history", 0); C["asst_in_history"] += int(any(m.get("role") == "assistant" for m in msgs))
        with LOCK: SEEN.append((hash(msgs[0]["content"]) if msgs else 0, len(msgs)))
        n = 40
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def chunk(d):
            s = f"data: {json.dumps(d)}\n\n".encode(); self.wfile.write(f"{len(s):x}\r\n".encode() + s + b"\r\n"); self.wfile.flush()
        base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": MODEL}
        chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        for i in range(n):
            time.sleep(0.002)
            with LOCK: C["drafts"] += 1; C["draft_tokens"] += 8; C["accepted"] += rng.choice([0, 1, 2, 3, 4])
            chunk({**base, "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}]})
        chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        chunk({**base, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": n, "total_tokens": 100 + n}})
        s = b"data: [DONE]\n\n"; self.wfile.write(f"{len(s):x}\r\n".encode() + s + b"\r\n0\r\n\r\n"); self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
