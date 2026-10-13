// Three.js view of the MuJoCo robot: one mesh per visual geom, posed from data.geom_xpos/xmat.
import * as THREE from 'three';

const MESH = 7; // mjGEOM_MESH
const VISUAL_GROUP = 2;

export function createRobotView(sim) {
  const { model, data } = sim;
  // MuJoCo is z-up; three is y-up. Everything MuJoCo lives under this group.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;

  const geoms = new Map(); // mesh id -> BufferGeometry
  const mats = new Map();
  const parts = [];

  function meshGeometry(id) {
    if (geoms.has(id)) return geoms.get(id);
    const va = model.mesh_vertadr[id], vn = model.mesh_vertnum[id];
    const fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
    const V = model.mesh_vert, F = model.mesh_face;
    const pos = new Float32Array(fn * 9);
    for (let f = 0; f < fn; f++) {
      for (let k = 0; k < 3; k++) {
        const v = F[(fa + f) * 3 + k];
        pos.set([V[(va + v) * 3], V[(va + v) * 3 + 1], V[(va + v) * 3 + 2]], f * 9 + k * 3);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.computeVertexNormals();
    geoms.set(id, g);
    return g;
  }

  function material(geom) {
    const mid = model.geom_matid[geom];
    const src = mid >= 0 ? model.mat_rgba : model.geom_rgba;
    const o = (mid >= 0 ? mid : geom) * 4;
    const key = [src[o], src[o + 1], src[o + 2], src[o + 3]].map(v => v.toFixed(3)).join();
    if (!mats.has(key)) {
      const dark = src[o] < 0.4;
      mats.set(key, new THREE.MeshStandardMaterial({
        color: new THREE.Color(src[o], src[o + 1], src[o + 2]),
        metalness: dark ? 0.2 : 0.55,
        roughness: dark ? 0.6 : 0.32,
      }));
    }
    return mats.get(key);
  }

  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_type[g] !== MESH || model.geom_group[g] !== VISUAL_GROUP) continue;
    const m = new THREE.Mesh(meshGeometry(model.geom_dataid[g]), material(g));
    m.castShadow = true;
    m.matrixAutoUpdate = false;
    world.add(m);
    parts.push([g, m]);
  }

  // Soft contact shadow under the pelvis: grounds the robot in a splat that has no real lighting.
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.55)');
  grad.addColorStop(0.5, 'rgba(0,0,0,0.25)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 0.9),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }),
  );
  shadow.position.z = 0.002; // MuJoCo frame: just above the floor
  world.add(shadow);

  const M = new THREE.Matrix4();
  function update() {
    const P = data.geom_xpos, R = data.geom_xmat;
    for (const [g, m] of parts) {
      const r = g * 9, p = g * 3;
      M.set(R[r], R[r + 1], R[r + 2], P[p],
            R[r + 3], R[r + 4], R[r + 5], P[p + 1],
            R[r + 6], R[r + 7], R[r + 8], P[p + 2],
            0, 0, 0, 1);
      m.matrix.copy(M);
    }
    const q = data.qpos;
    shadow.position.x = q[0];
    shadow.position.y = q[1];
    shadow.material.opacity = Math.max(0, Math.min(1, 1.6 - q[2] * 0.9));
  }

  // Pelvis position in three's (y-up) frame, for the follow camera.
  const v = new THREE.Vector3();
  function pelvis() {
    const q = data.qpos;
    return v.set(q[0], q[2], -q[1]);
  }

  update();
  return { world, update, pelvis };
}
