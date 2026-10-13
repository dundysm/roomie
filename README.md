# roomwalk

Drop in a Gaussian-splat scan of your room and drive a live Unitree G1 around it, all in the browser.

- **Physics:** MuJoCo (WebAssembly) at 200 Hz.
- **Control:** Unitree's own G1 velocity policy at 50 Hz. It's a small MLP run in plain JavaScript, with the same observations, action scales and PD gains it was trained with.
- **Room:** your scan, rendered with Spark. The floor is found automatically, and the scan becomes collision boxes, so the robot walks into your couch instead of through it.

## Run locally

```bash
python -m http.server 8770
# open http://localhost:8770, then drop in a .spz / .ply / .splat / .sog scan
```

Scan a room with a phone app that exports Gaussian splats (Scaniverse exports SPZ and PLY), keeping the floor in view. Controls: W/S walk, A/D strafe, Q/E turn, R reset, F follow camera; on touch screens, use the joystick.

## How it fits together

| File | What it does |
|---|---|
| `src/sim.js` | MuJoCo model, the 98-D observation, the policy MLP, PD targets, and 4 physics steps per policy step |
| `src/robot-view.js` | Three.js meshes posed from `geom_xpos` / `geom_xmat` |
| `src/obstacles.js` | Splat → occupancy grid (6 cm cells, 0.12–1.7 m above the floor) → merged MuJoCo boxes; picks an open spawn point |
| `src/app.js` | Scene, splat loading, automatic floor fit (RANSAC), 3-click floor fallback, driving, follow camera |
| `public/g1/` | `g1.xml` and meshes, `policy.bin` / `policy.json` |
| `tools/` | Exporters and the Python reference loop that the JS is checked against |

### Rebuilding the robot files

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe mjlab==1.2.0 onnx onnxruntime pyyaml
.venv/Scripts/python tools/export_g1.py <unitree_rl_mjlab checkout> public/g1   # training-exact MJCF
.venv/Scripts/python tools/export_policy.py public/g1                            # ONNX -> policy.bin/json
.venv/Scripts/python tools/sim_check.py public/g1 0.5 0 0 8                      # reference walk check
```

`export_policy.py` takes the action scales from the ONNX metadata, which holds the training values. `deploy.yaml` rounds the wrist scale to 0.07, but training used 0.075.

Reference loop results (8 s on a flat floor, achieved body velocity over the last 4 s):

| Command | Achieved | Fell |
|---|---|---|
| vx 0.5 | 0.495 m/s | no |
| vx 1.0 | 1.007 m/s | no |
| vy 0.4 | 0.318 m/s | no |
| vx 0.3, wz 0.8 | 0.295 m/s, wz 0.691 rad/s | no |
| stand | 0.000 | no |

## Credits and licences

- Robot model, meshes and velocity policy: [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) @ `1425b15`, Apache-2.0. `public/g1/g1.xml` is that model compiled by mjlab 1.2.0, with its actuators and armature.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Apache-2.0.
- [Spark](https://github.com/sparkjsdev/spark): MIT.
- [three.js](https://github.com/mrdoob/three.js): MIT.
