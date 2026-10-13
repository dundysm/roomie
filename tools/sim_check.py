"""Reference control loop for the mjlab G1 velocity policy; the browser port must match it.

Usage: .venv/Scripts/python tools/sim_check.py public/g1 [vx vy wz] [seconds]
Prints achieved base velocity over the last half of the run and whether the robot fell.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

D = Path(sys.argv[1])
CMD = np.array([float(x) for x in sys.argv[2:5]] if len(sys.argv) >= 5 else [0.5, 0.0, 0.0], dtype=np.float32)
SECS = float(sys.argv[5]) if len(sys.argv) >= 6 else 8.0

import json
cfg = json.load(open(D / "policy.json"))
DEFAULT = np.array(cfg["default_joint_pos"], dtype=np.float64)
SCALE = np.array(cfg["action_scale"], dtype=np.float64)
OFFSET = DEFAULT
RANGES = cfg["ranges"]
PERIOD = cfg["gait_period"]
STEP_DT = cfg["step_dt"]
BIN = np.fromfile(D / "policy.bin", dtype=np.float32)

def mlp(x):
    n = cfg["normalizer"]; k = cfg["obs_dim"]
    x = (x - BIN[n["mean"]:n["mean"] + k]) / BIN[n["std"]:n["std"] + k]
    for L in cfg["layers"]:
        w = BIN[L["w"]:L["w"] + L["in"] * L["out"]].reshape(L["out"], L["in"]); b = BIN[L["b"]:L["b"] + L["out"]]
        x = w @ x + b
        if L["elu"]: x = np.where(x > 0, x, np.expm1(x))
    return x

scene = f"""<mujoco><include file="g1.xml"/>
<worldbody><geom name="floor" type="plane" size="0 0 0.05" friction="1 0.005 0.0001"/></worldbody></mujoco>"""
(D / "scene_check.xml").write_text(scene)
m = mujoco.MjModel.from_xml_path(str(D / "scene_check.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
decim = round(STEP_DT / m.opt.timestep)

# Policy joint order = the robot's hinge joints in model order; actuators map onto joints by trnid.
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qadr = np.array([m.jnt_qposadr[j] for j in hinges])
vadr = np.array([m.jnt_dofadr[j] for j in hinges])
act_for = np.array([hinges.index(m.actuator_trnid[a, 0]) for a in range(m.nu)])

sess = ort.InferenceSession(str(Path(__file__).parent / "ref/policy.onnx"))
iname = sess.get_inputs()[0].name
last = np.zeros(29, dtype=np.float32)
phase = 0.0
cmd = np.array([np.clip(CMD[0], *RANGES["lin_vel_x"]), np.clip(CMD[1], *RANGES["lin_vel_y"]), np.clip(CMD[2], *RANGES["ang_vel_z"])], dtype=np.float32)

def obs():
    global phase
    quat = d.qpos[3:7]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    ang = d.qvel[3:6]                      # free-joint angular velocity is in the body frame
    grav = R.T @ np.array([0, 0, -1.0])
    phase = (phase + STEP_DT / PERIOD) % 1.0
    gp = [0.0, 0.0] if np.linalg.norm(cmd) < 0.1 else [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
    return np.concatenate([ang, grav, cmd, gp, d.qpos[qadr] - DEFAULT, d.qvel[vadr], last]).astype(np.float32)

steps = int(SECS / STEP_DT)
xs, fell = [], False
for k in range(steps):
    o = obs()
    a = mlp(o)
    ref = sess.run(None, {iname: o[None]})[0][0]
    maxdiff = max(globals().get("maxdiff", 0.0), float(np.abs(a - ref).max()))
    last = a.astype(np.float32)
    target = OFFSET + SCALE * a
    d.ctrl[:] = target[act_for]
    for _ in range(decim):
        mujoco.mj_step(m, d)
    if d.qpos[2] < 0.4:
        fell = True
        break
    if k >= steps // 2:
        R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[3:7]); R = R.reshape(3, 3)
        xs.append(R.T @ d.qvel[0:3])       # world linear velocity -> body frame
v = np.mean(xs, axis=0) if xs else np.zeros(3)
print(f"mlp vs onnx max|diff| {maxdiff:.2e}; obs dim {o.size}; cmd {cmd.tolist()} -> body vel {np.round(v, 3).tolist()}, wz {d.qvel[5]:.3f}; height {d.qpos[2]:.3f}; fell={fell}")
