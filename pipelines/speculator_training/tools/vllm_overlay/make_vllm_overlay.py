#!/usr/bin/env python
"""Build a vLLM overlay from a git commit: the installed `vllm` package as a symlink forest, with the Python files that the commit changes copied over it.

  python make_vllm_overlay.py --site <installed vllm package dir> --repo <vllm checkout> --out <NEW overlay dir> [--ref grug-dspark-layers]
  PYTHONPATH=<overlay dir> python ...        (the overlay dir goes first on the Python path)

Why an overlay: the Marin vLLM wheel in the env has no GrugDSparkModel yet. The change lives on the fork branch; until a wheel release carries it, an overlay applies exactly that
commit to the installed wheel. Properties that keep it honest:
  * the wheel's base commit is read from its dist-info version (`...+marin.<sha>`), and the ref must descend from it;
  * the files to overlay are the files `git diff <base> <ref>` changes (added or modified .py files under vllm/ only; anything else is refused, because an overlay cannot carry
    compiled or deleted files);
  * every modified file must be identical in the wheel and in the base commit, so the overlay cannot silently drop a change the wheel has;
  * files are read from the git commit, not from the working tree, so the overlay is the pushed commit and nothing else;
  * the overlay records its source commit in OVERLAY_SOURCE.txt and refuses to write into an existing directory.
"""
import argparse, datetime, glob, os, re, shutil, subprocess, sys


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, "-c", "core.preloadindex=false", *args], capture_output=True, check=check).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="installed vllm package directory (site-packages/vllm)")
    ap.add_argument("--repo", required=True, help="vllm git checkout")
    ap.add_argument("--out", required=True, help="NEW overlay directory")
    ap.add_argument("--ref", default="HEAD", help="commit or branch to overlay (default HEAD)")
    a = ap.parse_args()
    site, repo, out = (os.path.abspath(p) for p in (a.site, a.repo, a.out))
    for p in (site, repo):
        if not os.path.isdir(p):
            sys.exit(f"no such directory: {p}")
    if os.path.exists(out):
        sys.exit(f"{out} exists; overlays are never overwritten, pick a fresh directory")

    dist = glob.glob(os.path.join(os.path.dirname(site), "vllm-*.dist-info"))
    if len(dist) != 1:
        sys.exit(f"expected one vllm dist-info next to {site}, found {dist}")
    m = re.search(r"\+marin\.([0-9a-f]{7,40})", os.path.basename(dist[0]))
    if not m:
        sys.exit(f"cannot read the base commit from {os.path.basename(dist[0])} (expected '+marin.<sha>')")
    pin = m.group(1)
    ref_sha = git(repo, "rev-parse", "--verify", "--quiet", a.ref + "^{commit}", check=False).decode().strip()
    if not ref_sha:
        sys.exit(f"{a.ref} is not a commit or branch in {repo}")
    if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", pin, ref_sha]).returncode != 0:
        sys.exit(f"{a.ref} ({ref_sha[:10]}) does not descend from the wheel's base commit {pin}")

    changed = git(repo, "diff", "--name-status", "--no-renames", pin, ref_sha).decode().split("\n")
    files = []
    for line in filter(None, changed):
        status, path = line.split("\t", 1)
        if status not in ("A", "M") or not path.startswith("vllm/") or not path.endswith(".py"):
            sys.exit(f"cannot overlay '{status} {path}': only added or modified vllm/**/*.py files can be overlaid")
        files.append((status, path))
    if not files:
        sys.exit(f"{a.ref} has no changes relative to {pin}")
    for status, path in files:
        inst = os.path.join(site, path[len("vllm/"):])
        if status == "M":
            base = git(repo, "show", f"{pin}:{path}")
            if not os.path.exists(inst) or open(inst, "rb").read() != base:
                sys.exit(f"installed {path} differs from base {pin}: the overlay would drop the wheel's own change")
        elif os.path.exists(inst):
            sys.exit(f"{path} is new in {a.ref} but already exists in the wheel")
    paths = [p for _, p in files]

    def link(src, dst):
        """Real directories along the path of an overlaid file, symlinks for everything else."""
        rel = os.path.relpath(src, site)
        prefix = "vllm/" if rel == "." else f"vllm/{rel}/"
        os.makedirs(dst)
        for e in os.listdir(src):
            s, d = os.path.join(src, e), os.path.join(dst, e)
            if e == "__pycache__":
                continue
            if os.path.isdir(s) and any(p.startswith(f"{prefix}{e}/") for p in paths):
                link(s, d)
            else:
                os.symlink(s, d)

    link(site, os.path.join(out, "vllm"))
    for _, path in files:
        dst = os.path.join(out, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst):
            os.remove(dst)
        with open(dst, "wb") as f:
            f.write(git(repo, "show", f"{ref_sha}:{path}"))
    with open(os.path.join(out, "OVERLAY_SOURCE.txt"), "w") as f:
        f.write(f"ref: {a.ref}\ncommit: {ref_sha}\nwheel base commit: {pin}\nwheel dist-info: {os.path.basename(dist[0])}\n"
                f"built: {datetime.datetime.now().isoformat(timespec='seconds')}\nfiles:\n" + "".join(f"  {s} {p}\n" for s, p in files))
    print(f"overlay at {out}: {len(files)} files from {ref_sha[:10]} over wheel base {pin}")


if __name__ == "__main__":
    main()
