// Three.js view of the MuJoCo world: robot meshes and balls posed from data.geom_xpos/xmat.
import * as THREE from 'three';

const MESH = 7, SPHERE = 2; // mjtGeom
const VISUAL_GROUP = 2, BALL_GROUP = 1;

export function createRobotView(sim) {
  const { model, data } = sim;
  // MuJoCo is z-up; three is y-up. Everything MuJoCo lives under this group.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;

  const geoms = new Map(); // mesh id -> BufferGeometry
  const mats = new Map();
  const parts = [];        // [geom id, three mesh]
  const pickables = [];    // robot meshes, userData.body = MuJoCo body id

  function meshGeometry(id) {
    if (geoms.has(id)) return geoms.get(id);
    const va = model.mesh_vertadr[id];
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
    const key = [src[o], src[o + 1], src[o + 2]].map(v => v.toFixed(3)).join();
    if (!mats.has(key)) {
      const dark = src[o] < 0.4;
      mats.set(key, new THREE.MeshStandardMaterial({
        color: new THREE.Color(src[o], src[o + 1], src[o + 2]),
        metalness: dark ? 0.15 : 0.4,
        roughness: dark ? 0.5 : 0.32,
        envMapIntensity: 1.25,
      }));
    }
    return mats.get(key);
  }

  for (let g = 0; g < model.ngeom; g++) {
    const type = model.geom_type[g], group = model.geom_group[g];
    if (type === MESH && group === VISUAL_GROUP) {
      const m = new THREE.Mesh(meshGeometry(model.geom_dataid[g]), material(g));
      m.matrixAutoUpdate = false;
      m.userData.body = model.geom_bodyid[g];
      world.add(m);
      parts.push([g, m]);
      pickables.push(m);
    } else if (type === SPHERE && group === BALL_GROUP) {
      const r = model.geom_size[g * 3], c = model.geom_rgba;
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(r, 32, 16),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(c[g * 4], c[g * 4 + 1], c[g * 4 + 2]), roughness: 0.45, metalness: 0.0 }),
      );
      m.matrixAutoUpdate = false;
      world.add(m);
      parts.push([g, m]);
    }
  }

  // Soft contact shadows: robot and balls. A splat has no real lighting to cast into.
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.6)');
  grad.addColorStop(0.5, 'rgba(0,0,0,0.25)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  const shadowTex = new THREE.CanvasTexture(c);
  const mkShadow = size => {
    const s = new THREE.Mesh(new THREE.PlaneGeometry(size, size),
      new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
    s.position.z = 0.002; world.add(s); return s;
  };
  const shadow = mkShadow(0.9);
  const ballShadows = parts.filter(([g]) => model.geom_type[g] === SPHERE).map(([g]) => [g, mkShadow(0.35)]);

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
    for (const [g, s] of ballShadows) {
      s.position.x = P[g * 3]; s.position.y = P[g * 3 + 1];
      s.material.opacity = Math.max(0, 0.9 - P[g * 3 + 2] * 0.8);
    }
  }

  function setEnvMap(tex) {
    for (const m of mats.values()) { m.envMap = tex; m.needsUpdate = true; }
  }

  // Pelvis position in three's (y-up) frame, for the follow camera.
  const v = new THREE.Vector3();
  function pelvis() {
    const q = data.qpos;
    return v.set(q[0], q[2], -q[1]);
  }

  update();
  // The head: the robot mesh that sits highest above the pelvis (used for the robot's-eye camera).
  let headGeom = -1, best = -Infinity;
  for (const [g, m] of parts) {
    if (!m.userData.body) continue;
    const z = data.geom_xpos[g * 3 + 2] - data.qpos[2];
    if (z > best) { best = z; headGeom = g; }
  }
  return { world, update, pelvis, pickables, setEnvMap, headGeom };
}
