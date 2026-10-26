"""Applies draft-weight updates to the running server without making the trainer wait for a rollout batch.

Aurora's controller POSTs /update_weights_from_disk and gives up after 60 s. With the Datakit runs a rollout batch took a few seconds; with RLVR rollouts of up to 4,096 tokens one takes 80-100 s and
the update (which needs the rollout lock) timed out about half of the time. So the endpoint copies the trainer's draft directory (the trainer overwrites it in place at the next sync), answers at once, and a
background thread applies the NEWEST queued copy as soon as no rollout is running; an update that is still waiting when a newer one arrives is dropped."""
import shutil
import tempfile
import threading


class DraftUpdater:
    def __init__(self, apply_fn, lock, state, scratch_dir=None):
        self.apply_fn, self.lock, self.state, self.scratch_dir = apply_fn, lock, state, scratch_dir or None
        self.pending, self.pending_lock, self.wake = None, threading.Lock(), threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, model_path: str) -> None:
        d = tempfile.mkdtemp(prefix="draft_update_", dir=self.scratch_dir)
        shutil.copytree(model_path, d, dirs_exist_ok=True)
        with self.pending_lock:
            old, self.pending = self.pending, d
        if old:
            shutil.rmtree(old, ignore_errors=True)
        self.wake.set()

    def _loop(self):
        while True:
            self.wake.wait(); self.wake.clear()
            with self.pending_lock:
                d, self.pending = self.pending, None
            if d is None:
                continue
            try:
                with self.lock:
                    self.apply_fn(d)
                    self.state["draft_updates"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"draft update failed: {e!r}", flush=True)
                self.state["draft_update_errors"] = self.state.get("draft_update_errors", 0) + 1
            finally:
                shutil.rmtree(d, ignore_errors=True)
