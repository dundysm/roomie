// Find a scan's floor plane and a metric scale from its splat centres (splat-local coordinates).
//
// 1. Sequential RANSAC pulls out the dominant planes, with no orientation assumed.
// 2. Boundary planes (floor, ceiling, walls) have nearly the whole scene on one side; tabletops
//    and sofa seats don't. The floor is the best-supported boundary plane within ~50° of the
//    scan's ±Y (phone scans are Y-up; COLMAP-style scans are roughly Y-down, sometimes tilted).
// 3. A parallel boundary plane on the far side is a ceiling: scale so the gap is ROOM_H.
//    Without one, a scan whose top sits at a plausible metric height keeps scale 1; otherwise
//    the top of the scan is taken to be TOP_H.
const ROOM_H = 2.6, TOP_H = 2.3;

export function findFloor(splat) {
  const P = [];
  const stride = Math.max(1, Math.floor(splat.numSplats / 40000));
  splat.forEachSplat((i, c, sc, q, o) => { if (i % stride === 0 && o > 0.3) P.push(c.x, c.y, c.z); });
  const n = P.length / 3;
  if (n < 500) return null;
  const ext = [0, 1, 2].map(k => {
    const a = []; for (let i = 0; i < n; i++) a.push(P[i * 3 + k]);
    a.sort((x, y) => x - y);
    return a[Math.floor(n * 0.98)] - a[Math.floor(n * 0.02)];
  });
  const tol = 0.005 * Math.hypot(...ext);
  const used = new Uint8Array(n);
  const rnd = () => Math.floor(Math.random() * n);
  const planes = [];
  for (let round = 0; round < 6; round++) {
    let best = null;
    for (let it = 0; it < 600; it++) {
      const ia = rnd(), ib = rnd(), ic = rnd();
      if (used[ia] || used[ib] || used[ic]) continue;
      const a = ia * 3, b = ib * 3, c = ic * 3;
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz); if (l < 1e-12) continue;
      nx /= l; ny /= l; nz /= l;
      const d = -(nx * P[a] + ny * P[a + 1] + nz * P[a + 2]);
      let cnt = 0;
      for (let i = 0; i < n; i++) {
        if (used[i]) continue;
        if (Math.abs(nx * P[i * 3] + ny * P[i * 3 + 1] + nz * P[i * 3 + 2] + d) < tol) cnt++;
      }
      if (!best || cnt > best.cnt) best = { cnt, nx, ny, nz, d };
    }
    if (!best) break;
    let pos = 0, neg = 0;
    for (let i = 0; i < n; i++) {
      const e = best.nx * P[i * 3] + best.ny * P[i * 3 + 1] + best.nz * P[i * 3 + 2] + best.d;
      if (Math.abs(e) < tol) used[i] = 1; else if (e > 0) pos++; else neg++;
    }
    const s = pos >= neg ? 1 : -1; // orient each plane so the scene is on its + side
    planes.push({ nx: best.nx * s, ny: best.ny * s, nz: best.nz * s, d: best.d * s,
                  support: best.cnt / n, oneSide: Math.max(pos, neg) / Math.max(1, pos + neg) });
  }
  const boundary = planes.filter(p => p.oneSide >= 0.9);
  const cands = boundary.filter(p => Math.abs(p.ny) >= 0.64);
  if (!cands.length) return null;
  // floor vs ceiling: more support wins; furniture sits on floors, so count points just above
  const nearAbove = p => {
    let k = 0;
    for (let i = 0; i < n; i++) {
      const h = p.nx * P[i * 3] + p.ny * P[i * 3 + 1] + p.nz * P[i * 3 + 2] + p.d;
      if (h > 4 * tol && h < 0.25 * Math.max(...ext)) k++;
    }
    return k / n;
  };
  cands.forEach(p => { p.score = p.support + nearAbove(p); });
  cands.sort((a, b) => b.score - a.score);
  const F = cands[0];
  const height = i => F.nx * P[i * 3] + F.ny * P[i * 3 + 1] + F.nz * P[i * 3 + 2] + F.d;

  let scale = 1, how = 'metric';
  const ceil = boundary.find(p => p !== F && (p.nx * F.nx + p.ny * F.ny + p.nz * F.nz) < -0.95);
  if (ceil) {
    const gap = Math.abs(F.d + ceil.d); // opposite normals: the two offsets add to the separation
    scale = ROOM_H / gap; how = 'ceiling';
  } else {
    const hs = []; for (let i = 0; i < n; i++) { const h = height(i); if (h > 0) hs.push(h); }
    hs.sort((a, b) => a - b);
    const top = hs[Math.floor(hs.length * 0.98)] || TOP_H;
    if (top < 1.4 || top > 4) { scale = TOP_H / top; how = 'scan height'; }
  }
  return {
    n: [F.nx, F.ny, F.nz],
    c: [-F.d * F.nx, -F.d * F.ny, -F.d * F.nz], // the scan origin dropped onto the floor
    scale, how, support: F.support,
  };
}
