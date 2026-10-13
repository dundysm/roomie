import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { createSim } from './sim.js';
import { createRobotView } from './robot-view.js';
import { buildObstacles, obstacleView } from './obstacles.js';

const $ = id => document.getElementById(id);
const status = msg => { $('status').textContent = msg; };

// ---------- three + spark ----------
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const spark = new SparkRenderer({ renderer });
scene.add(spark);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.02, 200);
camera.position.set(1.8, 1.4, 2.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.7, 0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x4a4036, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2, 4, 1.5);
scene.add(sun);

const grid = new THREE.GridHelper(20, 40, 0x2a3240, 0x1a2029);
scene.add(grid);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- room (the user's scan) ----------
const room = new THREE.Group();
scene.add(room);
let splat = null, sceneKey = null;
let calib = null; // { n:[3], c:[3] } floor plane in splat-local coordinates
let roomScale = 1;

function applyCalib() {
  if (!calib) { room.position.set(0, 0, 0); room.quaternion.identity(); room.scale.setScalar(roomScale); return; }
  const n = new THREE.Vector3(...calib.n), c = new THREE.Vector3(...calib.c);
  const q = new THREE.Quaternion().setFromUnitVectors(n, new THREE.Vector3(0, 1, 0));
  room.quaternion.copy(q);
  room.scale.setScalar(roomScale);
  room.position.copy(c.multiplyScalar(roomScale).applyQuaternion(q).negate());
  room.updateMatrixWorld(true);
}

function saveCalib() {
  if (!sceneKey) return;
  try { localStorage.setItem('roomwalk:' + sceneKey, JSON.stringify({ calib, roomScale })); } catch {}
}
function loadCalib() {
  calib = null; roomScale = 1;
  try {
    const s = JSON.parse(localStorage.getItem('roomwalk:' + sceneKey) || 'null');
    if (s) { calib = s.calib; roomScale = s.roomScale || 1; }
  } catch {}
  $('scale').value = Math.log2(roomScale);
  applyCalib();
}

// Find the floor in splat-local coordinates: RANSAC over splat centres for the horizontal plane
// (normal near the scan's ±Y) with the most support, oriented so most of the scene lies above it.
function autoFloor(s) {
  const P = [];
  const stride = Math.max(1, Math.floor(s.numSplats / 40000));
  s.forEachSplat((i, c, sc, q, o) => { if (i % stride === 0 && o > 0.3) P.push(c.x, c.y, c.z); });
  const n = P.length / 3;
  if (n < 500) return null;
  const ys = []; for (let i = 0; i < n; i++) ys.push(P[i * 3 + 1]);
  ys.sort((a, b) => a - b);
  const tol = Math.max(0.01, 0.008 * (ys[Math.floor(n * 0.98)] - ys[Math.floor(n * 0.02)]));
  const rnd = () => Math.floor(Math.random() * n);
  let best = null;
  for (let it = 0; it < 400; it++) {
    const a = rnd() * 3, b = rnd() * 3, c = rnd() * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz); if (l < 1e-9) continue;
    nx /= l; ny /= l; nz /= l;
    if (Math.abs(ny) < 0.92) continue;
    const d = -(nx * P[a] + ny * P[a + 1] + nz * P[a + 2]);
    let cnt = 0, above = 0;
    for (let i = 0; i < n * 3; i += 3) {
      const e = nx * P[i] + ny * P[i + 1] + nz * P[i + 2] + d;
      if (Math.abs(e) < tol) cnt++; else if (e > 0) above++;
    }
    if (!best || cnt > best.cnt) best = { cnt, nx, ny, nz, d, above };
  }
  if (!best) return null;
  let { nx, ny, nz, d } = best;
  if (best.above < n - best.cnt - best.above) { nx = -nx; ny = -ny; nz = -nz; d = -d; }
  // Spawn where the scan was captured from: the origin, dropped onto the floor plane.
  return { n: [nx, ny, nz], c: [-d * nx, -d * ny, -d * nz], support: best.cnt / n };
}

async function loadSplat(opts, key, label) {
  status(`loading ${label}…`);
  if (splat) { room.remove(splat); splat.dispose?.(); }
  splat = new SplatMesh(opts);
  room.add(splat);
  await splat.initialized;
  sceneKey = key;
  loadCalib();
  let how = 'floor restored';
  if (!calib) {
    calib = autoFloor(splat);
    how = calib ? 'floor found automatically · wrong? click “set floor”' : null;
    if (calib) { applyCalib(); saveCalib(); }
  }
  grid.visible = !calib;
  if (!calib) { status(`${label} loaded · click 3 points on the floor`); startFloorPick(); return; }
  rebuildWorld();
  resetRobot(); frameRobot();
  status(`${label} · ${how}`);
}

$('file').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (f) loadSplat({ fileBytes: await f.arrayBuffer(), fileName: f.name }, `${f.name}:${f.size}`, f.name);
});
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', async e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) loadSplat({ fileBytes: await f.arrayBuffer(), fileName: f.name }, `${f.name}:${f.size}`, f.name);
});

// ---------- picking: floor calibration and robot placement ----------
const raycaster = new THREE.Raycaster();
let mode = null, picks = [];
const markers = new THREE.Group();
scene.add(markers);

function ndc(e) {
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
function startFloorPick() {
  mode = 'floor'; picks = []; markers.clear();
  status('click 3 points spread across the floor');
}
$('floor').onclick = () => splat ? startFloorPick() : status('load a scan first');
$('place').onclick = () => { mode = 'place'; status('click a spot on the floor to drop the robot there'); };

let down = null;
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!mode || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; // ignore drags
  raycaster.setFromCamera(ndc(e), camera);
  if (mode === 'floor') {
    const hit = raycaster.intersectObject(splat, false)[0];
    if (!hit) return status('missed the scan; click on the floor');
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.025), new THREE.MeshBasicMaterial({ color: 0xffd34d }));
    dot.position.copy(hit.point); markers.add(dot);
    picks.push(room.worldToLocal(hit.point.clone()));
    if (picks.length < 3) return status(`point ${picks.length}/3 · click ${3 - picks.length} more`);
    const [a, b, c] = picks;
    const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).normalize();
    const cen = a.clone().add(b).add(c).multiplyScalar(1 / 3);
    const camLocal = room.worldToLocal(camera.position.clone());
    if (n.dot(camLocal.sub(cen)) < 0) n.negate(); // the floor's up side faces the camera
    calib = { n: n.toArray(), c: cen.toArray() };
    applyCalib(); saveCalib(); markers.clear(); mode = null; grid.visible = false;
    rebuildWorld();
    resetRobot(); frameRobot();
    status('floor set · drive with WASD, Q/E to turn');
  } else if (mode === 'place') {
    const p = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p)) return;
    resetRobot(p.x, -p.z); mode = null; status('placed');
  }
});

// ---------- robot ----------
let sim, view;
function facingYaw() { // face away from the camera, in MuJoCo's frame
  const d = new THREE.Vector3(); camera.getWorldDirection(d);
  return Math.atan2(-d.z, d.x);
}
function resetRobot(x, y) {
  if (!sim) return;
  const sp = obstacles?.spawn;
  if (x === undefined) sim.reset(sp ? sp.x : 0, sp ? sp.y : 0, sp ? sp.yaw : facingYaw());
  else sim.reset(x, y, facingYaw());
  view.update();
  $('fell').hidden = true;
}
// Clear distance from (x, y) along heading th over open floor, in MuJoCo coordinates.
function clearRun(x, y, th, max) {
  if (!obstacles) return max;
  let s = 0;
  while (s < max && obstacles.free(x + Math.cos(th) * s, y + Math.sin(th) * s)) s += 0.05;
  return s;
}
function frameRobot() {
  const p = view.pelvis(), q = sim.data.qpos;
  const yaw = 2 * Math.atan2(q[6], q[3]);
  // look from behind the robot if there's room, else from whichever side has the most
  let best = { th: yaw + Math.PI, run: -1 };
  for (const off of [0, 0.5, -0.5, 1, -1, 1.6, -1.6, Math.PI]) {
    const th = yaw + Math.PI + off, run = clearRun(q[0], q[1], th, 2.2);
    if (run > best.run + 0.15) best = { th, run };
  }
  const back = Math.max(0.9, Math.min(2.0, best.run - 0.25));
  controls.target.copy(p);
  camera.position.set(q[0] + Math.cos(best.th) * back, 1.45, -(q[1] + Math.sin(best.th) * back));
  lastPelvis.copy(p);
}
$('reset').onclick = () => resetRobot();

// ---------- driving ----------
const keys = new Set();
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  keys.add(e.code);
  if (e.code === 'KeyR') resetRobot();
  if (e.code === 'KeyF') toggleFollow();
});
addEventListener('keyup', e => keys.delete(e.code));
const joy = { x: 0, y: 0 };
{
  const pad = $('joy'), knob = $('knob');
  let id = null;
  const move = e => {
    const r = pad.getBoundingClientRect(), R = r.width / 2;
    let dx = e.clientX - r.left - R, dy = e.clientY - r.top - R;
    const l = Math.hypot(dx, dy); if (l > R) { dx *= R / l; dy *= R / l; }
    knob.style.transform = `translate(${dx}px, ${dy}px)`;
    joy.x = dx / R; joy.y = -dy / R;
  };
  pad.addEventListener('pointerdown', e => { id = e.pointerId; pad.setPointerCapture(id); move(e); });
  pad.addEventListener('pointermove', e => { if (e.pointerId === id) move(e); });
  const end = () => { id = null; joy.x = joy.y = 0; knob.style.transform = ''; };
  pad.addEventListener('pointerup', end); pad.addEventListener('pointercancel', end);
}
function targetCmd() {
  const k = c => keys.has(c) ? 1 : 0;
  let vx = (k('KeyW') + k('ArrowUp')) - (k('KeyS') + k('ArrowDown'));
  let vy = k('KeyA') - k('KeyD');
  let wz = (k('KeyQ') + k('ArrowLeft')) - (k('KeyE') + k('ArrowRight'));
  vx = vx > 0 ? 0.8 * vx : 0.4 * vx; vy *= 0.4; wz *= 0.9;
  if (joy.x || joy.y) { vx = joy.y > 0 ? joy.y * 0.9 : joy.y * 0.45; wz = -joy.x * 1.0; }
  return [vx, vy, wz];
}

// ---------- follow camera ----------
let follow = true;
const lastPelvis = new THREE.Vector3();
function toggleFollow() { follow = !follow; $('follow').classList.toggle('on', follow); }
$('follow').onclick = toggleFollow;

let scaleT = 0;
$('scale').addEventListener('input', e => {
  roomScale = 2 ** +e.target.value; applyCalib(); saveCalib();
  clearTimeout(scaleT); scaleT = setTimeout(() => { rebuildWorld(); resetRobot(); }, 350);
});

// ---------- room collisions ----------
let obstacles = null, showColliders = false;
function rebuildWorld() {
  if (!sim) return;
  obstacles = splat && calib ? buildObstacles(splat, room) : null;
  sim.rebuild(obstacles ? obstacles.xml : '');
  scene.remove(view.world);
  view = createRobotView(sim);
  scene.add(view.world);
  if (showColliders && obstacles) view.world.add(obstacleView(obstacles));
  $('colliders').textContent = obstacles ? `colliders (${obstacles.boxes.length})` : 'colliders';
}
$('colliders').onclick = () => {
  showColliders = !showColliders;
  $('colliders').classList.toggle('on', showColliders);
  rebuildWorld();
};

// ---------- main loop ----------
let acc = 0, prev = performance.now();
function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
  if (sim) {
    const t = targetCmd();
    for (let i = 0; i < 3; i++) sim.cmd[i] += (t[i] - sim.cmd[i]) * Math.min(1, dt * 6);
    acc += dt;
    let n = 0;
    while (acc >= sim.cfg.step_dt && n++ < 5) { sim.step(); acc -= sim.cfg.step_dt; }
    if (n >= 5) acc = 0;
    view.update();
    if (sim.fallen) $('fell').hidden = false;
    if (follow) {
      const p = view.pelvis(), d = p.clone().sub(lastPelvis);
      camera.position.add(d); controls.target.add(d); lastPelvis.copy(p);
      // don't let the camera sit inside a wall or furniture: ease it toward the robot
      if (obstacles && !obstacles.free(camera.position.x, -camera.position.z)) {
        const toR = p.clone().sub(camera.position); toR.y = 0;
        if (toR.length() > 0.8) camera.position.addScaledVector(toR, Math.min(1, dt * 4));
      }
    }
    const v = sim.bodyVel();
    $('hud').textContent = `cmd ${sim.cmd[0].toFixed(2)} m/s · turn ${sim.cmd[2].toFixed(2)} rad/s   |   actual ${v[0].toFixed(2)} m/s`;
  }
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(loop);
window.__rw = { get splat() { return splat; }, room, camera, controls, get sim() { return sim; } };

// ---------- boot ----------
(async () => {
  status('loading physics and policy…');
  sim = await createSim('public/g1/');
  view = createRobotView(sim);
  scene.add(view.world);
  frameRobot();
  status('drop in a room scan (.spz / .ply / .splat / .sog) · or just drive: WASD, Q/E');
  const q = new URLSearchParams(location.search).get('scene');
  if (q) loadSplat({ url: q }, q, q.split('/').pop());
})().catch(err => { console.error(err); status('failed to start: ' + err.message); });
