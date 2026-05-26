#!/usr/bin/env python3
"""Local single-worker drain of MIWN's ree_K3 dd queue.

For each ready task: claim (locally), invoke numerics/ree_K3/solve.py at
dps=32 to ||F||<1e-20, mark done with the allocated version + metrics,
commit.  Loops until time-budget exhausted or queue drained.

usage: python3 numerics/ree_K3/drain.py [--budget 540]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/user/MIWN")
QUEUE = ROOT / "todo" / "TASK_QUEUE.json"
REG = ROOT / "solutions" / "REGISTRY.json"


def load():
    return json.loads(QUEUE.read_text())


def save(q):
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")


def git(*args):
    subprocess.run(["git", "-C", str(ROOT), *args], check=False, capture_output=True)


def commit(msg):
    git("add", "todo/TASK_QUEUE.json", "solutions/")
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", msg],
                   check=False, capture_output=True)


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=540)
    ap.add_argument("--per-task", type=int, default=500)
    ap.add_argument("--G", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    done = 0
    fail = 0
    while time.time() - t0 < a.budget:
        q = load()
        nxt = next((t for t in q["tasks"] if t["status"] == "ready"), None)
        if not nxt:
            print("[drain] queue drained", flush=True)
            break
        tid = nxt["id"]; gamma = nxt["gamma"]; tau = nxt["tau"]
        nxt["status"] = "claimed"; nxt["claimed_by"] = "local-drain"; nxt["claimed_at"] = utcnow()
        save(q)
        budget_left = a.budget - (time.time() - t0)
        ttl = max(60, min(a.per_task, int(budget_left) - 10))
        print(f"[drain] start {tid} g={gamma} tau={tau} (ttl {ttl}s)", flush=True)
        t1 = time.time()
        r = subprocess.run(
            ["python3", "numerics/ree_K3/solve.py", "--gamma", str(gamma),
             "--tau", str(tau), "--G", str(a.G)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=ttl,
        )
        dt = time.time() - t1
        if r.returncode != 0:
            print(f"[drain] FAIL {tid} rc={r.returncode} ({dt:.0f}s)", flush=True)
            print(r.stderr[-600:] or r.stdout[-600:], flush=True)
            q = load()
            for t in q["tasks"]:
                if t["id"] == tid:
                    t["status"] = "ready"; t["claimed_by"] = None; t["claimed_at"] = None
            save(q); commit(f"ree_K3: release {tid} after failure")
            fail += 1
            continue
        reg = json.loads(REG.read_text()) if REG.exists() else {"solutions": []}
        rezn = [s for s in reg.get("solutions", []) if s.get("problem") == "ree_K3"]
        last = rezn[-1] if rezn else {"version": "v????", "one_minus_R2": None, "F_inf": None}
        q = load()
        for t in q["tasks"]:
            if t["id"] == tid:
                t["status"] = "done"
                t["checkpoint"] = f"solutions/pool/ree_K3/{last['version']}/"
                t["result"] = {"one_minus_R2": last.get("one_minus_R2"),
                               "F_inf": last.get("F_inf")}
                t["completed_at"] = utcnow()
        save(q)
        commit(f"ree_K3: solve {tid} -> {last['version']} 1-R2={last.get('one_minus_R2')}")
        print(f"[drain] DONE {tid} {last['version']} 1-R2={last.get('one_minus_R2')} F={last.get('F_inf')} ({dt:.0f}s)", flush=True)
        done += 1
    print(f"[drain] summary: {done} solved, {fail} failed in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
