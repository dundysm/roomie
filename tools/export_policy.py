"""Flatten the mjlab G1 velocity policy (ONNX) into policy.bin + policy.json for a dependency-free JS MLP.

Usage: .venv/Scripts/python tools/export_policy.py public/g1
The normalizer, MLP layers and the training-time constants all come from the ONNX file itself
(its metadata carries the exact action scales; deploy.yaml rounds the wrist scale to 0.07).
"""
import json, sys
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import numpy_helper

D = Path(sys.argv[1])
m = onnx.load(Path(__file__).parent / "ref/policy.onnx")
W = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in m.graph.initializer}
meta = {p.key: p.value for p in m.metadata_props}
ops = [n.op_type for n in m.graph.node]
assert ops == ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"], ops
div = next(n for n in m.graph.node if n.op_type == "Div").input[1]

blobs, layers, off = [], [], 0
def add(a):
    global off
    a = np.ascontiguousarray(a.ravel(), dtype=np.float32); blobs.append(a); o = off; off += a.size; return o
mean, std = add(W["obs_normalizer._mean"]), add(W[div])
for i in (0, 2, 4, 6):
    w, b = W[f"mlp.{i}.weight"], W[f"mlp.{i}.bias"]
    layers.append({"out": w.shape[0], "in": w.shape[1], "w": add(w), "b": add(b), "elu": i != 6})
np.concatenate(blobs).tofile(D / "policy.bin")

f = lambda k: [float(x) for x in meta[k].split(",")]
cfg = yaml.safe_load(open(Path(__file__).parent / "ref/deploy.yaml"))
out = {
    "source": "unitreerobotics/unitree_rl_mjlab@1425b15 deploy/robots/g1/config/policy/velocity/v0 (Apache-2.0)",
    "joint_names": meta["joint_names"].split(","),
    "default_joint_pos": f("default_joint_pos"),
    "action_scale": f("action_scale"),
    "obs_dim": int(W["obs_normalizer._mean"].shape[1]),
    "normalizer": {"mean": mean, "std": std},
    "layers": layers,
    "step_dt": cfg["step_dt"],
    "gait_period": cfg["observations"]["gait_phase"]["params"]["period"],
    "ranges": cfg["commands"]["base_velocity"]["ranges"],
    "obs_order": list(cfg["observations"].keys()),
}
json.dump(out, open(D / "policy.json", "w"), indent=1)
print(f"policy.bin {off} floats; layers {[(l['in'], l['out']) for l in layers]}")
