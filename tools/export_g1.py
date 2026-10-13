"""Export the G1 exactly as unitree_rl_mjlab trains it (actuators, armature, collisions) to plain MJCF.

Usage: .venv/Scripts/python tools/export_g1.py <unitree_rl_mjlab checkout> <out dir>
Writes <out>/g1.xml (meshes referenced as assets/<file>) and copies the meshes it uses.
"""
import shutil, sys
from pathlib import Path

import mujoco

MJLAB = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(MJLAB))

from mjlab.entity.entity import Entity  # noqa: E402
from src.assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg  # noqa: E402

robot = Entity(get_g1_robot_cfg())
spec = robot.spec

# Same solver settings as the velocity task's SimulationCfg.
spec.option.timestep = 0.005
spec.option.iterations = 10
spec.option.ls_iterations = 20

model = spec.compile()
print(f"nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt} ngeom={model.ngeom}")
names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
acts = [model.actuator(i).name for i in range(model.nu)]
print("joints:", names)
print("actuators match joints:", acts == names or [model.actuator(i).trnid[0] for i in range(model.nu)])

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "assets").mkdir(exist_ok=True)
spec.meshdir = "assets"
xml = spec.to_xml()
(OUT / "g1.xml").write_text(xml)
src_assets = MJLAB / "src/assets/robots/unitree_g1/xmls/assets"
used = {m.file for m in spec.meshes}
for f in used:
    shutil.copy(src_assets / Path(f).name, OUT / "assets" / Path(f).name)
print(f"wrote {OUT/'g1.xml'} and {len(used)} meshes")
