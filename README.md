# center of mass

**A live humanoid in your 3D scan.** Drop in a Gaussian-splat scan of your room and a live Unitree G1 walks around it, in your browser.

**Live demo:** https://dundysm.github.io/center-of-mass/

The robot is not animated:
- **Physics:** MuJoCo (WebAssembly) runs at 200 Hz.
- **Control:** Unitree's own G1 walking policy runs at 50 Hz, with the observations, action scales and PD gains it was trained with.
- **The room becomes physics:** the floor is found automatically, and the scan is turned into collision geometry.
- **Interaction:** shove the robot, throw balls at it, send it across the room, or look through its head camera.

## What you can do

- **Walk it:** WASD, the arrow keys or the touch joystick. Movement is camera-relative, so the robot turns to face the way you push. Hold shift to run.
- **Send it somewhere:** click the floor. It plans a path around your furniture (A* on the scan's collision grid) and walks there.
- **Shove it:** drag any part of the robot. It's pulled by a spring force and fights to stay on its feet; if it falls, it gets back up.
- **Throw things at it:** press B or the "throw ball" button. Balls bounce off your real furniture.
- **See what it sees:** press V for its head-camera view, rendered from your scan.
- **Record a clip:** use the rec button to save a video of the canvas.

It runs at 60 fps on a laptop iGPU (Intel Arc 140V).

## Bring your own room

1. Scan a room with [Scaniverse](https://scaniverse.com) (free on iOS and Android). Walk slowly around the room for about a minute, keeping the floor in view.
2. Process the scan as a **splat** and export **SPZ** (PLY also works).
3. Drop the file onto the page.

Floor and scale are set automatically; `src/floor.js` handles both metric phone scans and tilted research scans in arbitrary units. If either comes out wrong, use the set-floor button (click 3 floor points) or the scale slider. Your scan never leaves your browser.

## Run locally

```bash
python -m http.server 8770
# open http://localhost:8770 and drop in a scan
```

There's no build step: plain ES modules, with three.js, Spark and MuJoCo loaded from jsDelivr.

## How it fits together

| File | What it does |
|---|---|
| `src/sim.js` | MuJoCo model, the 98-D observation, the policy MLP, PD targets, and 4 physics steps per policy step |
| `src/robot-view.js` | Three.js meshes posed from `geom_xpos` / `geom_xmat` |
| `src/obstacles.js` | Splat → occupancy grid (6 cm cells, 0.12–1.7 m above the floor) → merged MuJoCo boxes; open spawn point; A* planner |
| `src/floor.js` | Floor plane and scale: sequential RANSAC, boundary-plane test, floor vs ceiling by furniture mass |
| `src/app.js` | Scene, splat loading, steering, click-to-walk, shove and throw, env-map lighting, head camera, recording |
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

- This project's own code: Apache-2.0 (`LICENSE`).

- Robot model, meshes and velocity policy: [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) @ `1425b15`, Apache-2.0. `public/g1/g1.xml` is that model compiled by mjlab 1.2.0, with its actuators and armature.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Apache-2.0.
- [Spark](https://github.com/sparkjsdev/spark): MIT.
- [three.js](https://github.com/mrdoob/three.js): MIT.