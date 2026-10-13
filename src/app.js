import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { createSim } from './sim.js';
import { createRobotView } from './robot-view.js';
import { buildObstacles, obstacleView } from './obstacles.js';
import { findFloor } from './floor.js';

const $ = id => document.getElementById(id);
const status = msg => { $('status').textContent = msg; };
// MuJoCo (z-up) <-> three (y-up)
const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);
const toMj = v => [v.x, -v.z, v.y];

// ---------- three + spark ----------
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
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

const hemi = new THREE.HemisphereLight(0xffffff, 0x4a4036, 1.4);
scene.add(hemi);
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
    calib = findFloor(splat);
    how = calib ? `floor found · scale from ${calib.how}` : null;
    if (calib) { roomScale = calib.scale; $('scale').value = Math.log2(roomScale); applyCalib(); saveCalib(); }
  }
  grid.visible = !calib;
  if (!calib) { status(`${label} loaded · click 3 points on the floor`); startFloorPick(); return; }
  rebuildWorld();
  resetRobot(); frameRobot();
  status(`${label} · ${how} · drag the robot to shove it, B to throw a ball`);
}
async function loadFile(f) { if (f) loadSplat({ fileBytes: await f.arrayBuffer(), fileName: f.name }, `${f.name}:${f.size}`, f.name); }
$('file').addEventListener('change', e => loadFile(e.target.files[0]));
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => { e.preventDefault(); loadFile(e.dataTransfer.files[0]); });

// ---------- robot ----------
let sim, view, envMap = null;
function facingYaw() { // face away from the camera, in MuJoCo's frame
  const d = new THREE.Vector3(); camera.getWorldDirection(d);
  return Math.atan2(-d.z, d.x);
}
function resetRobot(x, y) {
  if (!sim) return;
  const sp = obstacles?.spawn;
  if (x === undefined) sim.reset(sp ? sp.x : 0, sp ? sp.y : 0, sp ? sp.yaw : facingYaw());
  else sim.reset(x, y, facingYaw(), true);
  view.update(); clearGoal(); fallenAt = 0;
  $('fell').hidden = true;
  refreshEnvMap(true);
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

// The robot reflects the actual room: render a cube map of the splat from where the robot
// stands, refreshed when it has walked a metre or so.
const envRenderer = new SparkRenderer({ renderer });
let envAt = null, envBusy = false;
async function refreshEnvMap(force = false) {
  if (!splat || !view || envBusy) return;
  const p = view.pelvis().clone().setY(1.0);
  if (!force && envAt && envAt.distanceTo(p) < 1.2) return;
  envBusy = true;
  try {
    const tex = await envRenderer.renderEnvMap({
      scene, worldCenter: p,
      hideObjects: [view.world, goalRing, pathLine, grid, markers, grabLine, pipGroup].filter(Boolean),
    });
    envMap = tex; envAt = p; view.setEnvMap(tex);
    hemi.intensity = 0.75; sun.intensity = 1.0;
  } catch (e) { console.warn('env map', e); }
  envBusy = false;
}

// ---------- room collisions ----------
let obstacles = null, showColliders = false;
function rebuildWorld() {
  if (!sim) return;
  obstacles = splat && calib ? buildObstacles(splat, room) : null;
  sim.rebuild(obstacles ? obstacles.xml : '');
  scene.remove(view.world);
  view = createRobotView(sim);
  scene.add(view.world);
  if (envMap) view.setEnvMap(envMap);
  if (showColliders && obstacles) view.world.add(obstacleView(obstacles));
  $('colliders').textContent = obstacles ? `colliders (${obstacles.boxes.length})` : 'colliders';
}
$('colliders').onclick = () => { showColliders = !showColliders; $('colliders').classList.toggle('on', showColliders); rebuildWorld(); };
let scaleT = 0;
$('scale').addEventListener('input', e => {
  roomScale = 2 ** +e.target.value; applyCalib(); saveCalib();
  clearTimeout(scaleT); scaleT = setTimeout(() => { rebuildWorld(); resetRobot(); }, 350);
});

// ---------- pointer: floor pick, place, shove, click-to-walk ----------
const raycaster = new THREE.Raycaster();
let mode = null, picks = [];
const markers = new THREE.Group();
scene.add(markers);
const mouse = new THREE.Vector2();
function ndc(e) {
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
function startFloorPick() { mode = 'floor'; picks = []; markers.clear(); status('click 3 points spread across the floor'); }
$('floor').onclick = () => splat ? startFloorPick() : status('load a scan first');
$('place').onclick = () => { mode = 'place'; status('click a spot on the floor to drop the robot there'); };
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

// Shove: grab any part of the robot and pull it with a spring; it fights to stay upright.
let grab = null;
const grabLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0xffd34d, depthTest: false }));
grabLine.visible = false; grabLine.renderOrder = 11; scene.add(grabLine);

let down = null;
renderer.domElement.addEventListener('pointerdown', e => {
  down = [e.clientX, e.clientY];
  if (mode || !sim || e.button !== 0) return;
  raycaster.setFromCamera(ndc(e), camera);
  const hit = raycaster.intersectObjects(view.pickables, false)[0];
  if (!hit) return;
  const body = hit.object.userData.body;
  const n = new THREE.Vector3(); camera.getWorldDirection(n);
  grab = { body, plane: new THREE.Plane().setFromNormalAndCoplanarPoint(n, hit.point), target: hit.point.clone(), prev: null };
  controls.enabled = false; clearGoal();
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch {}
});
renderer.domElement.addEventListener('pointermove', e => {
  mouse.copy(ndc(e));
  if (!grab) return;
  raycaster.setFromCamera(mouse, camera);
  raycaster.ray.intersectPlane(grab.plane, grab.target);
});
renderer.domElement.addEventListener('pointerup', e => {
  if (grab) { grab = null; sim.push = null; grabLine.visible = false; controls.enabled = true; return; }
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; // ignore drags
  raycaster.setFromCamera(ndc(e), camera);
  if (!mode) { // click-to-walk
    const p = new THREE.Vector3();
    if (sim && raycaster.ray.intersectPlane(floorPlane, p)) setGoal(p.x, -p.z);
    return;
  }
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
    if (n.dot(camLocal.sub(cen)) < 0) n.negate();
    calib = { n: n.toArray(), c: cen.toArray() };
    applyCalib(); saveCalib(); markers.clear(); mode = null; grid.visible = false;
    rebuildWorld(); resetRobot(); frameRobot();
    status('floor set');
  } else if (mode === 'place') {
    const p = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(floorPlane, p)) return;
    resetRobot(p.x, -p.z); mode = null; status('placed');
  }
});
function applyGrab(dt) {
  if (!grab) return;
  const d = sim.data, b = grab.body * 3;
  const com = toThree(d.xipos[b], d.xipos[b + 1], d.xipos[b + 2]);
  const vel = grab.prev ? com.clone().sub(grab.prev).divideScalar(Math.max(dt, 1e-3)) : new THREE.Vector3();
  grab.prev = com.clone();
  const f = grab.target.clone().sub(com).multiplyScalar(700).addScaledVector(vel, -60);
  if (f.length() > 450) f.setLength(450);
  sim.push = { body: grab.body, force: toMj(f) };
  grabLine.geometry.setFromPoints([com, grab.target]); grabLine.visible = true;
}

// ---------- throwing ----------
function throwBall() {
  if (!sim) return;
  raycaster.setFromCamera(mouse, camera);
  const dir = raycaster.ray.direction.clone();
  const from = camera.position.clone().addScaledVector(dir, 0.35);
  const v = dir.multiplyScalar(9).add(new THREE.Vector3(0, 1.2, 0));
  sim.throwBall(toMj(from), toMj(v));
}
$('throw').onclick = () => { mouse.set(0, 0); throwBall(); };

// ---------- driving ----------
const keys = new Set();
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  keys.add(e.code);
  if (e.code === 'KeyR') resetRobot();
  if (e.code === 'KeyF') toggleFollow();
  if (e.code === 'KeyB' && !e.repeat) throwBall();
  if (e.code === 'KeyV') togglePip();
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

// Click-to-walk plans a path around the furniture (A* on the collision grid) and follows it.
let goal = null, path = [];
const goalRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.2, 40),
  new THREE.MeshBasicMaterial({ color: 0xffd34d, transparent: true, opacity: 0.9, depthTest: false }));
goalRing.rotation.x = -Math.PI / 2; goalRing.visible = false; goalRing.renderOrder = 10;
scene.add(goalRing);
const pathLine = new THREE.Line(new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xffd34d, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.85, depthTest: false }));
pathLine.visible = false; pathLine.renderOrder = 10; scene.add(pathLine);
function drawPath() {
  const q = sim.data.qpos;
  const pts = [[q[0], q[1]], ...path].map(([x, y]) => new THREE.Vector3(x, 0.02, -y));
  pathLine.geometry.setFromPoints(pts); pathLine.computeLineDistances(); pathLine.visible = pts.length > 1;
}
function setGoal(x, y) {
  const q = sim.data.qpos;
  const planned = obstacles ? obstacles.plan([q[0], q[1]], [x, y]) : null;
  path = planned && planned.length ? planned : [[x, y]];
  const [gx, gy] = path[path.length - 1];
  goal = { x: gx, y: gy };
  goalRing.position.set(gx, 0.01, -gy); goalRing.visible = true;
  drawPath();
  if (obstacles && !planned) status("can't find a way there; walking straight");
}
function clearGoal() { goal = null; path = []; goalRing.visible = false; pathLine.visible = false; }

// Third-person steering: a heading controller turns the robot toward the input direction (keys or
// joystick, relative to the camera) or the next path waypoint, and walks once it's roughly facing it.
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
function targetCmd() {
  const k = c => keys.has(c) ? 1 : 0;
  let fwd = k('KeyW') + k('ArrowUp') - k('KeyS') - k('ArrowDown');
  let side = k('KeyD') + k('ArrowRight') - k('KeyA') - k('ArrowLeft');
  let mag = Math.min(1, Math.hypot(fwd, side));
  if (joy.x || joy.y) { fwd = joy.y; side = joy.x; mag = Math.min(1, Math.hypot(fwd, side)); }
  const run = keys.has('ShiftLeft') || keys.has('ShiftRight');
  const q = sim.data.qpos, yaw = 2 * Math.atan2(q[6], q[3]);
  let heading = null, speed = 0;
  if (grab) return [0, 0, 0];
  if (mag > 0.05) {
    clearGoal();
    const f = new THREE.Vector3(); camera.getWorldDirection(f); f.y = 0; f.normalize();
    const mx = f.x * fwd - f.z * side, mz = f.z * fwd + f.x * side;
    heading = Math.atan2(-mz, mx);
    speed = mag * (run ? 1.0 : 0.65);
  } else if (path.length) {
    let [wx, wy] = path[0];
    let d = Math.hypot(wx - q[0], wy - q[1]);
    if (path.length > 1 && d < 0.3) { path.shift(); [wx, wy] = path[0]; d = Math.hypot(wx - q[0], wy - q[1]); }
    if (path.length === 1 && d < 0.22) { clearGoal(); return [0, 0, 0]; }
    heading = Math.atan2(wy - q[1], wx - q[0]);
    const left = path.reduce((s, p, i) => s + (i ? Math.hypot(p[0] - path[i - 1][0], p[1] - path[i - 1][1]) : d), 0);
    speed = Math.min(run ? 1.0 : 0.65, 0.25 + left * 0.6);
    drawPath();
  }
  if (heading === null) return [0, 0, 0];
  const err = wrap(heading - yaw);
  const wz = Math.max(-2, Math.min(2, 2.6 * err));
  const vx = Math.abs(err) > 1.3 ? 0 : speed * Math.max(0, Math.cos(err)) ** 2;
  return [vx, 0, wz];
}

// ---------- robot's-eye view (picture in picture) ----------
const PIP_W = 384, PIP_H = 216;
const headCam = new THREE.PerspectiveCamera(80, PIP_W / PIP_H, 0.05, 100);
const pipRenderer = new SparkRenderer({ renderer, target: { width: PIP_W, height: PIP_H, doubleBuffer: true } });
const pipScene = new THREE.Scene(), pipCam = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
const pipGroup = new THREE.Group(); pipScene.add(pipGroup);
const pipQuad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: new THREE.Texture() }));
const pipFrame = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
pipGroup.add(pipFrame, pipQuad);
let pip = false;
function togglePip() { pip = !pip; $('pip').classList.toggle('on', pip); }
$('pip').onclick = togglePip;
function renderPip() {
  if (!pip || !view) return;
  const d = sim.data, g = view.headGeom, R = d.geom_xmat, P = d.geom_xpos;
  const fwd = toThree(R[g * 9], R[g * 9 + 3], R[g * 9 + 6]).normalize();
  const pos = toThree(P[g * 3], P[g * 3 + 1], P[g * 3 + 2]).addScaledVector(fwd, 0.13).add(new THREE.Vector3(0, 0.02, 0));
  headCam.position.copy(pos);
  headCam.lookAt(pos.clone().add(fwd).add(new THREE.Vector3(0, -0.25, 0)));
  pipRenderer.renderTarget({ scene, camera: headCam });
  pipQuad.material.map = pipRenderer.target.texture; pipQuad.material.needsUpdate = true;
  // place the quad in pixels, bottom-left
  const W = innerWidth, H = innerHeight, w = Math.min(PIP_W, W * 0.42), h = w * PIP_H / PIP_W, m = 14;
  pipCam.right = W; pipCam.top = H; pipCam.updateProjectionMatrix();
  pipQuad.scale.set(w, h, 1); pipQuad.position.set(m + w / 2, m + 56 + h / 2, 0);
  pipFrame.scale.set(w + 4, h + 4, 1); pipFrame.position.set(m + w / 2, m + 56 + h / 2, -0.5);
  renderer.autoClear = false; renderer.clearDepth(); renderer.render(pipScene, pipCam); renderer.autoClear = true;
}

// ---------- record a clip ----------
let rec = null;
$('rec').onclick = () => {
  if (rec) { rec.stop(); return; }
  const stream = renderer.domElement.captureStream(30);
  const type = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t));
  const chunks = [];
  rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8e6 });
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(chunks, { type }));
    a.download = `roomwalk-${Date.now()}.${type.includes('mp4') ? 'mp4' : 'webm'}`;
    a.click(); rec = null; $('rec').classList.remove('on'); $('rec').textContent = 'rec';
  };
  rec.start(); $('rec').classList.add('on'); $('rec').textContent = '■ stop';
};

// ---------- follow camera ----------
let follow = true;
const lastPelvis = new THREE.Vector3();
function toggleFollow() { follow = !follow; $('follow').classList.toggle('on', follow); }
$('follow').onclick = toggleFollow;
function followCam(dt) {
  const p = view.pelvis(), d = p.clone().sub(lastPelvis);
  camera.position.add(d); controls.target.add(d); lastPelvis.copy(p);
  // Third-person boom: keep 1.3–3.5 m from the robot; if the camera ends up inside a wall or
  // furniture, pull it in (never closer than 1.3 m) and lift it to look over the obstacle.
  const off = camera.position.clone().sub(p), flat = Math.hypot(off.x, off.z);
  const setFlat = k => { camera.position.x = p.x + off.x * k; camera.position.z = p.z + off.z * k; };
  if (obstacles && obstacles.solid(camera.position.x, -camera.position.z)) {
    if (flat > 1.3) setFlat(Math.max(1.3 / flat, 1 - dt * 3));
    camera.position.y = Math.min(p.y + 1.1, camera.position.y + dt * 1.0);
  } else if (camera.position.y > p.y + 0.7) camera.position.y -= dt * 0.4; // settle back down
  if (flat < 1.3 && flat > 1e-3) setFlat(1.3 / flat);
  else if (flat > 3.5) setFlat(3.5 / flat);
}

// ---------- main loop ----------
let acc = 0, prev = performance.now(), fallenAt = 0;
function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
  if (sim) {
    const t = targetCmd();
    for (let i = 0; i < 3; i++) sim.cmd[i] += (t[i] - sim.cmd[i]) * Math.min(1, dt * 8);
    applyGrab(dt);
    acc += dt;
    let n = 0;
    while (acc >= sim.cfg.step_dt && n++ < 5) { sim.step(); acc -= sim.cfg.step_dt; }
    if (n >= 5) acc = 0;
    view.update();
    if (sim.fallen) {
      $('fell').hidden = false;
      fallenAt ||= now;
      if (now - fallenAt > 2500) { const q = sim.data.qpos; resetRobot(q[0], q[1]); } // get back up
    }
    if (follow) followCam(dt);
    refreshEnvMap();
    const v = sim.bodyVel();
    $('hud').textContent = `${v[0].toFixed(2)} m/s · ${sim.cmd[2].toFixed(1)} rad/s` + (grab ? ' · shoving' : '');
  }
  controls.update();
  renderer.render(scene, camera);
  renderPip();
}
requestAnimationFrame(loop);
window.__rw = { get splat() { return splat; }, room, camera, controls, get sim() { return sim; }, get goal() { return goal; }, get path() { return path; }, get obstacles() { return obstacles; }, get view() { return view; } };

// ---------- boot ----------
(async () => {
  status('loading physics and policy…');
  sim = await createSim('public/g1/');
  view = createRobotView(sim);
  scene.add(view.world);
  frameRobot();
  status('drop in a room scan (.spz / .ply / .splat / .sog) · or just drive: WASD');
  const q = new URLSearchParams(location.search).get('scene');
  if (q) loadSplat({ url: q }, q, q.split('/').pop());
})().catch(err => { console.error(err); status('failed to start: ' + err.message); });
