"""A/B probe: plan() only, eager backend (no CUDA graph, no cuDNN branch).

Prints the decoded ABC plan for comparison against the cuda_graph run.
Offline mode avoids HF HEAD-request retries since weights are already cached.
"""
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from yue2 import YuE2Pipeline

request = json.loads(open("examples/waltz.json", encoding="utf-8").read())

with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae",
                                  device="cuda", backend=sys.argv[1] if len(sys.argv) > 1 else "torch") as pipe:
    plan = pipe.plan(**request)
    print("=== ABC PLAN (torch-eager) ===")
    print(repr(plan.abc))
    print("truncated:", plan.truncated)