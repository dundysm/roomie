// Turn a calibrated splat into coarse MuJoCo collision boxes: an occupancy grid over the floor
// of everything between ankle and head height, merged into runs along x.
import * as THREE from 'three';

const CELL = 0.06;          // m
const ZMIN = 0.12, ZMAX = 1.7;
const CLEAR_R = 0.45;       // keep the spawn point free

export function buildObstacles(splat, room) {
  room.updateMatrixWorld(true);
  const M = room.matrixWorld, v = new THREE.Vector3();
  const stride = Math.max(1, Math.floor(splat.numSplats / 300000));
  const cells = new Map(); // "i,j" -> [count, maxZ]
  const floor = new Map();  // "i,j" -> splats lying on the floor: where the room actually is
  splat.forEachSplat((i, c, sc, q, o) => {
    if (i % stride || o < 0.4) return;
    v.copy(c).applyMatrix4(M);
    const x = v.x, y = -v.z, z = v.y; // three (y-up) -> MuJoCo (z-up)
    const ci = Math.floor(x / CELL), cj = Math.floor(y / CELL), k = ci + ',' + cj;
    if (Math.abs(z) < 0.06) { floor.set(k, (floor.get(k) || 0) + 1); return; }
    if (z < ZMIN || z > ZMAX) return;
    const e = cells.get(k);
    if (e) { e[0]++; if (z > e[1]) e[1] = z; } else cells.set(k, [1, z]);
  });
  const need = Math.max(2, Math.round(6 / stride));
  const occ = new Set();
  for (const [k, [n]] of cells) if (n >= need) occ.add(k);
  // floor coverage, closed by one cell so small holes in the scan don't split the room
  const onFloor = new Set();
  for (const k of floor.keys()) {
    const [i, j] = k.split(',').map(Number);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) onFloor.add((i + a) + ',' + (j + b));
  }
  const spawn = findSpawn(occ, onFloor);
  const rows = new Map(); // j -> sorted occupied i with heights
  for (const [k, [n, zmax]] of cells) {
    if (n < need) continue;
    const [i, j] = k.split(',').map(Number);
    const cx = (i + 0.5) * CELL, cy = (j + 0.5) * CELL;
    if (Math.hypot(cx - spawn.x, cy - spawn.y) < CLEAR_R) continue;
    if (!rows.has(j)) rows.set(j, []);
    rows.get(j).push([i, Math.min(ZMAX, zmax)]);
  }
  const boxes = [];
  for (const [j, list] of rows) {
    list.sort((a, b) => a[0] - b[0]);
    let s = null;
    for (const [i, h] of list) {
      const hb = Math.ceil(h / 0.25) * 0.25; // bucket heights so runs merge
      if (s && i === s.i1 + 1 && hb === s.h) { s.i1 = i; continue; }
      if (s) boxes.push(s);
      s = { j, i0: i, i1: i, h: hb };
    }
    if (s) boxes.push(s);
  }
  const xml = boxes.map(b => {
    const hx = (b.i1 - b.i0 + 1) * CELL / 2, cx = b.i0 * CELL + hx, cy = (b.j + 0.5) * CELL;
    return `<geom type="box" pos="${cx.toFixed(3)} ${cy.toFixed(3)} ${(b.h / 2).toFixed(3)}" size="${hx.toFixed(3)} ${(CELL / 2).toFixed(3)} ${(b.h / 2).toFixed(3)}" rgba="0 0 0 0" condim="3" friction="0.8"/>`;
  }).join('\n');
  // open floor test in MuJoCo coordinates, for keeping the camera out of walls and furniture
  const free = (x, y) => {
    const k = Math.floor(x / CELL) + ',' + Math.floor(y / CELL);
    return onFloor.has(k) && !occ.has(k);
  };
  // inside a wall or furniture (unscanned floor doesn't count): for the camera
  const solid = (x, y) => occ.has(Math.floor(x / CELL) + ',' + Math.floor(y / CELL));
  return { xml, boxes, cell: CELL, spawn, free, solid, plan: planner(occ, onFloor) };
}

// A* over the floor grid with obstacles inflated by the robot's half-width; the result is
// shortcut by line of sight so the robot walks straight segments, not a staircase.
const INFLATE = 0.3;
function planner(occ, onFloor) {
  const key = (i, j) => i + ',' + j;
  const r = Math.ceil(INFLATE / CELL), blocked = new Set();
  for (const k of occ) {
    const [i, j] = k.split(',').map(Number);
    for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) if (a * a + b * b <= r * r) blocked.add(key(i + a, j + b));
  }
  const ok = (i, j) => onFloor.has(key(i, j)) && !blocked.has(key(i, j));
  const cellOf = (x, y) => [Math.floor(x / CELL), Math.floor(y / CELL)];
  const center = ([i, j]) => [(i + 0.5) * CELL, (j + 0.5) * CELL];
  function nearestOk([i, j]) {
    for (let d = 0; d < 25; d++)
      for (let a = -d; a <= d; a++) for (let b = -d; b <= d; b++)
        if (Math.max(Math.abs(a), Math.abs(b)) === d && ok(i + a, j + b)) return [i + a, j + b];
    return null;
  }
  function los(a, b) { // straight line stays on walkable cells
    const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2);
    for (let s = 0; s <= n; s++) {
      const t = s / Math.max(1, n);
      if (!ok(Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t))) return false;
    }
    return true;
  }
  return function plan(from, to) {
    const s = nearestOk(cellOf(...from)), g = nearestOk(cellOf(...to));
    if (!s || !g) return null;
    const open = [[0, s]], gs = new Map([[key(...s), 0]]), prev = new Map();
    const h = c => Math.hypot(c[0] - g[0], c[1] - g[1]);
    let found = false, iters = 0;
    while (open.length && iters++ < 60000) {
      let bi = 0; for (let k = 1; k < open.length; k++) if (open[k][0] < open[bi][0]) bi = k;
      const [, c] = open.splice(bi, 1)[0];
      if (c[0] === g[0] && c[1] === g[1]) { found = true; break; }
      const gc = gs.get(key(...c));
      for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const n = [c[0] + a, c[1] + b], k = key(...n);
        if (!ok(...n)) continue;
        const ng = gc + Math.hypot(a, b);
        if (ng < (gs.get(k) ?? Infinity)) { gs.set(k, ng); prev.set(k, c); open.push([ng + h(n), n]); }
      }
    }
    if (!found) return null;
    const path = [g];
    for (let k = key(...g); prev.has(k); k = key(...prev.get(k))) path.push(prev.get(k));
    path.reverse();
    const out = [path[0]];
    for (let i = 1; i < path.length; i++) if (!los(out[out.length - 1], path[i])) out.push(path[i - 1]);
    out.push(path[path.length - 1]);
    return out.slice(1).map(center);
  };
}

// The most open floor cell reachable from where the scan was captured (the origin), plus the
// direction with the longest clear run, so the robot starts in the room facing into it.
function findSpawn(occ, onFloor) {
  const R = Math.round(4 / CELL), key = (i, j) => i + ',' + j;
  // multi-source BFS distance (in cells) from obstacles, over a window around the origin
  const dist = new Map(), q = [];
  for (const k of occ) {
    const [i, j] = k.split(',').map(Number);
    if (Math.abs(i) <= R + 20 && Math.abs(j) <= R + 20) { dist.set(k, 0); q.push([i, j]); }
  }
  for (let h = 0; h < q.length; h++) {
    const [i, j] = q[h], d = dist.get(key(i, j));
    if (d >= 30) continue;
    for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const k = key(i + a, j + b);
      if (!dist.has(k) && Math.abs(i + a) <= R + 20 && Math.abs(j + b) <= R + 20) { dist.set(k, d + 1); q.push([i + a, j + b]); }
    }
  }
  const clear = (i, j) => dist.has(key(i, j)) ? dist.get(key(i, j)) : 30;
  // flood the free region connected to the origin (bounded), keep the clearest cell
  const walkable = (i, j) => !occ.has(key(i, j)) && onFloor.has(key(i, j));
  let start = [0, 0];
  outer: for (let r = 0; r < R; r++) { // nearest walkable cell to the capture origin
    for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) {
      if (Math.max(Math.abs(a), Math.abs(b)) === r && walkable(a, b)) { start = [a, b]; break outer; }
    }
  }
  const seen = new Set([key(...start)]), f = [start];
  let best = { i: start[0], j: start[1], c: -1 };
  for (let h = 0; h < f.length; h++) {
    const [i, j] = f[h], c = clear(i, j) - 0.02 * Math.hypot(i, j);
    if (c > best.c) best = { i, j, c };
    for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const ni = i + a, nj = j + b, k = key(ni, nj);
      if (seen.has(k) || !walkable(ni, nj) || Math.abs(ni) > R || Math.abs(nj) > R) continue;
      seen.add(k); f.push([ni, nj]);
    }
  }
  let yaw = 0, run = -1;
  for (let a = 0; a < 16; a++) {
    const th = a * Math.PI / 8, dx = Math.cos(th), dy = Math.sin(th);
    let s = 0;
    while (s < 80 && walkable(Math.round(best.i + dx * s), Math.round(best.j + dy * s))) s++;
    if (s > run) { run = s; yaw = th; }
  }
  return { x: (best.i + 0.5) * CELL, y: (best.j + 0.5) * CELL, yaw, clearance: clear(best.i, best.j) * CELL };
}

// Translucent view of the boxes, in the MuJoCo frame (put it under the robot view's `world`).
export function obstacleView({ boxes, cell }) {
  const g = new THREE.Group();
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshBasicMaterial({ color: 0x4dd0ff, transparent: true, opacity: 0.22, depthWrite: false });
  const inst = new THREE.InstancedMesh(geo, mat, Math.max(1, boxes.length));
  const m = new THREE.Matrix4();
  boxes.forEach((b, k) => {
    const sx = (b.i1 - b.i0 + 1) * cell;
    m.makeScale(sx, cell, b.h).setPosition(b.i0 * cell + sx / 2, (b.j + 0.5) * cell, b.h / 2);
    inst.setMatrixAt(k, m);
  });
  inst.count = boxes.length;
  g.add(inst);
  return g;
}
