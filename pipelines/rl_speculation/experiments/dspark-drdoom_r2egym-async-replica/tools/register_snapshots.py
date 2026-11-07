#!/usr/bin/env python3
"""Register the reference setup's 11 per-repo R2E-Gym images as Daytona snapshots in the org of DAYTONA_API_KEY.

Names (harbor__<env hash>__snapshot) come from the reference setup's images.json; images are the public
<registry prefix>/<repo>@sha256:<digest> (prefix passed with --registry-prefix) (digests from the v3-<recipe sha12> tags). Never deletes or modifies an
existing snapshot: an existing name is only reported. Resources match the RL arms (1 CPU, 2 GB RAM, 10 GB disk).
usage: DAYTONA_API_KEY=... register_snapshots.py --images images.json --digests ghcr_digests.json --registry-prefix ghcr.io/<owner>/r2egym [--only aiohttp] [--dry]
"""
import argparse, asyncio, json, os, sys, time
from daytona import AsyncDaytona, CreateSnapshotParams, DaytonaConfig, Resources

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--digests", required=True)
    ap.add_argument("--registry-prefix", required=True, help="image registry path before /<repo>, e.g. ghcr.io/<owner>/r2egym"); ap.add_argument("--only", default=""); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    imgs = json.load(open(a.images))["images"]
    dig = {r[0]: r[2] for r in json.load(open(a.digests))}
    todo = [i for i in imgs if not a.only or i["repo"] in a.only.split(",")]
    async with AsyncDaytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"])) as d:
        have = {s.name: s for s in (await d.snapshot.list(page=1, limit=100)).items}
        print(f"org has {len(have)} snapshots before", flush=True)
        for i in sorted(todo, key=lambda x: x["size_gb"]):
            name, repo = i["name"], i["repo"]
            if name in have:
                print(f"{repo}: {name} exists ({have[name].state}); skipped", flush=True); continue
            ref = f"{a.registry_prefix.rstrip('/')}/{repo}@{dig[repo]}"
            print(f"{repo}: creating {name} from {ref}", flush=True)
            if a.dry: continue
            t0 = time.time()
            await d.snapshot.create(CreateSnapshotParams(name=name, image=ref, resources=Resources(cpu=1, memory=2, disk=10)))
            while True:
                s = await d.snapshot.get(name)
                if "ACTIVE" in str(s.state): print(f"{repo}: ACTIVE after {time.time()-t0:.0f} s", flush=True); break
                if any(k in str(s.state) for k in ("ERROR", "BUILD_FAILED")): print(f"{repo}: FAILED {s.state} {s.error_reason}", flush=True); sys.exit(1)
                if time.time() - t0 > a.timeout: print(f"{repo}: timeout, left as is ({s.state})", flush=True); sys.exit(1)
                await asyncio.sleep(10)
        print(f"org has {len((await d.snapshot.list(page=1, limit=100)).items)} snapshots after", flush=True)
asyncio.run(main())
