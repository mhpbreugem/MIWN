"""Generate todo/TASK_QUEUE.json for a (gamma,tau) grid (>=100 cases).

Preserves existing tasks (status/checkpoint/result) and only ADDS new (gamma,tau)
points as `ready`.  Task id: ree_K3_g{round(gamma*100):04d}_t{round(tau*100):04d}.
"""
import json, datetime, os

QPATH = "todo/TASK_QUEUE.json"

GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.6, 2.5, 4.0, 6.5, 10.0]
TAUS   = [0.3, 0.5, 0.8, 1.0, 1.6, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 15.0]

def tid(g, t):
    return f"ree_K3_g{round(g*100):04d}_t{round(t*100):04d}"

def main():
    q = json.load(open(QPATH))
    tasks = q["tasks"]
    by_id = {t["id"]: t for t in tasks}
    added = 0
    for g in GAMMAS:
        for t in TAUS:
            i = tid(g, t)
            if i in by_id:
                continue
            tasks.append({
                "id": i, "problem": "ree_K3", "status": "ready",
                "gamma": g, "tau": t, "dps": 32, "depends_on": [],
                "checkpoint": None, "result": None,
                "claimed_by": None, "claimed_at": None, "completed_at": None,
                "note": "grid expansion (gamma,tau) sweep >=100 cases",
            })
            added += 1
    from collections import Counter
    st = Counter(t.get("status") for t in tasks)
    q["tasks"] = tasks
    q["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    q["summary"] = {"n_tasks": len(tasks), **{k: st[k] for k in st}}
    json.dump(q, open(QPATH, "w"), indent=2)
    print(f"added {added} new tasks; total {len(tasks)}; status {dict(st)}")

if __name__ == "__main__":
    main()
