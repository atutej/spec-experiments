import fcntl, os, time

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
