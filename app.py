"""
SIEVE demo server.

    python3 app.py            then open http://127.0.0.1:5000

One page, one button.  The button runs the real engine on the real cached PJM
lattice and streams every permutation back as it lands, so what is on screen is
a live run and not a replay.  Nothing here touches the network.

Two devices
-----------
The page offers CPU and, where one exists, GPU.  Both consume the SAME
permutations (see backends.py), so the device changes how long the run takes
and nothing else.  `test_backends.py` asserts that equality.

The lattice loads once at import, not per request: building the Problem reads
387 MB and computes the observed statistic for all 66,844 rules.
"""
from __future__ import annotations

import json
import queue
import threading

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, render_template, request, send_from_directory

import backends
import engine
from engine import Problem, load_cached

# the corrected PJM series calibrates at 504h; see FINAL.md
engine.BLOCK = 504

app = Flask(__name__)

print("loading lattice ...", flush=True)
_rules, _labels, _y, _X = load_cached()
PROB = Problem(_rules, _labels, _y, months=pd.to_datetime(_X.index).month.values,
               device="cpu")
_BACKENDS = {"cpu": PROB.backend}
HAS_GPU = backends.gpu_available()
print(f"ready: {PROB.n:,} hours, {len(PROB.labels):,} rules, gpu={HAS_GPU}", flush=True)

_lock = threading.Lock()

try:
    CANON = np.load("results2.npz")["nb"]
except Exception:
    CANON = None


def use_device(name):
    """Swap the backend in place.  Built once per device, then cached."""
    if name not in _BACKENDS:
        _BACKENDS[name] = backends.make(name, PROB.rules)
    PROB.backend = _BACKENDS[name]
    return PROB.backend


@app.route("/")
def index():
    canon = None
    if CANON is not None:
        canon = dict(J=int(CANON.size), n=int(len(PROB.significant(CANON))))
    devices = [dict(id="cpu", label="This Mac", sub=_BACKENDS["cpu"].label)]
    if HAS_GPU:
        try:
            g = use_device("gpu"); use_device("cpu")
            devices.append(dict(id="gpu", label="RunPod", sub=g.label))
        except Exception as e:
            print("gpu init failed:", e, flush=True)
    return render_template(
        "index.html",
        n=PROB.n, n1=PROB.n1, base=round(100 * float(PROB.y.mean()), 1),
        rules=len(PROB.labels), testable=int(PROB.s.testable.sum()),
        pruned=round(100 * (1 - float(PROB.s.testable.mean())), 2),
        observed=float(PROB.lp.min()), devices=devices, canon=canon)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/run")
def run():
    J = max(10, min(int(request.args.get("perms", 200)), 2000))
    seed = int(request.args.get("seed", 1))
    dev = request.args.get("device", "cpu")
    if dev not in ("cpu", "gpu") or (dev == "gpu" and not HAS_GPU):
        dev = "cpu"

    def stream():
        q: queue.Queue = queue.Queue()

        def work():
            try:
                with _lock:                       # one run at a time
                    b = use_device(dev)
                    q.put(("meta", dict(observed=float(PROB.lp.min()), J=J,
                                        device=dev, label=b.label)))
                    nulls = {}
                    for kind in ("block", "row"):
                        import time
                        t0 = time.time()

                        def cb(j, JJ, v, kind=kind):
                            q.put(("perm", dict(kind=kind, j=j, J=JJ, v=v)))

                        nul = PROB.null(J, kind=kind, seed=seed, cb=cb)
                        nulls[kind] = nul
                        est, sd = PROB.count_ci(nul)
                        q.put(("null_done", dict(
                            kind=kind, n=int(len(PROB.significant(nul))),
                            est=est, sd=sd, cutoff=PROB.cutoff(nul),
                            ms=1000 * (time.time() - t0) / J)))
                    # findings always come from the BLOCK null; the row shuffle
                    # is run only to show that it fails
                    chosen, cum = PROB.cover(nulls["block"], top=6)
                    q.put(("findings", dict(rules=chosen, covered=cum,
                                            total=int(PROB.n1))))
                    q.put(("done", {}))
            except Exception as e:
                q.put(("error", dict(msg=f"{type(e).__name__}: {e}")))
            q.put((None, None))

        threading.Thread(target=work, daemon=True).start()
        while True:
            kind, payload = q.get()
            if kind is None:
                break
            yield _sse(dict(type=kind, **payload))

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/fig/<name>")
def fig(name):
    import os
    if not name.endswith(".png") or "/" in name or ".." in name:
        abort(404)
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), name)


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)
