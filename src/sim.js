// MuJoCo + the unitree_rl_mjlab G1 velocity policy, ported 1:1 from tools/sim_check.py,
// plus the things you can do to the robot: throw balls at it and push it around.
import loadMujoco from '@mujoco/mujoco';

const HINGE = 3, FREE = 0; // mjtJoint
export const N_BALLS = 8;
const BALL_R = 0.1;
const BALL_COLORS = ['1 0.36 0.3', '1 0.8 0.25', '0.3 0.8 1', '0.55 0.95 0.45', '1 0.45 0.8', '0.95 0.95 0.95', '1 0.55 0.15', '0.6 0.5 1'];

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

class Policy {
  constructor(cfg, bin) {
    this.cfg = cfg;
    this.w = bin;
    const n = cfg.normalizer, k = cfg.obs_dim;
    this.mean = bin.subarray(n.mean, n.mean + k);
    this.std = bin.subarray(n.std, n.std + k);
    this.bufs = cfg.layers.map(L => new Float32Array(L.out));
    this.x0 = new Float32Array(k);
  }
  act(obs) {
    let x = this.x0;
    for (let i = 0; i < x.length; i++) x[i] = (obs[i] - this.mean[i]) / this.std[i];
    this.cfg.layers.forEach((L, li) => {
      const y = this.bufs[li], W = this.w, wo = L.w, bo = L.b;
      for (let o = 0; o < L.out; o++) {
        let s = W[bo + o];
        const row = wo + o * L.in;
        for (let i = 0; i < L.in; i++) s += W[row + i] * x[i];
        y[o] = L.elu ? (s > 0 ? s : Math.expm1(s)) : s;
      }
      x = y;
    });
    return x; // the last layer's buffer: valid until the next call
  }
}

export async function createSim(base) {
  const mujoco = await loadMujoco();
  const [xmlRaw, cfg, bin] = await Promise.all([
    fetch(base + 'g1.xml').then(r => r.text()),
    fetch(base + 'policy.json').then(r => r.json()),
    fetch(base + 'policy.bin').then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
  ]);
  // The robot file's keyframe has the robot-only qpos size; the scene adds balls, so reset by hand.
  const xml = xmlRaw.replace(/<keyframe>[\s\S]*?<\/keyframe>/, '');
  const FS = mujoco.FS;
  FS.mkdir('/g1'); FS.mkdir('/g1/assets');
  FS.writeFile('/g1/g1.xml', xml);
  const meshFiles = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map(m => m[1]))];
  await Promise.all(meshFiles.map(async f => FS.writeFile('/g1/assets/' + f, await fetchBytes(base + 'assets/' + f))));

  const balls = Array.from({ length: N_BALLS }, (_, i) =>
    `<body name="ball${i}" pos="${100 + i} 100 ${BALL_R}"><freejoint/>` +
    `<geom type="sphere" size="${BALL_R}" mass="0.4" group="1" rgba="${BALL_COLORS[i]} 1" condim="3" friction="0.7 0.01 0.002" solref="0.006 0.25"/></body>`).join('\n');

  const policy = new Policy(cfg, bin);
  const baseScene = new TextDecoder().decode(await fetchBytes(base + 'scene.xml'));
  let model, data, hinges, qadr, vadr, actFor, decim, ballQ = [], ballV = [], robotBodies = 0;
  function compile(extraWorld = '') {
    FS.writeFile('/g1/scene.xml', baseScene.replace('</worldbody>', extraWorld + '\n' + balls + '\n  </worldbody>'));
    const m = mujoco.MjModel.from_xml_path('/g1/scene.xml');
    if (data) data.delete();
    if (model) model.delete();
    model = m; data = new mujoco.MjData(model);
    hinges = []; ballQ = []; ballV = [];
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] === HINGE) hinges.push(j);
      else if (model.jnt_type[j] === FREE && j > 0) { ballQ.push(model.jnt_qposadr[j]); ballV.push(model.jnt_dofadr[j]); }
    }
    robotBodies = model.nbody - N_BALLS; // world + robot bodies come first
    qadr = hinges.map(j => model.jnt_qposadr[j]);
    vadr = hinges.map(j => model.jnt_dofadr[j]);
    actFor = [];
    for (let a = 0; a < model.nu; a++) actFor.push(hinges.indexOf(model.actuator_trnid[2 * a]));
    decim = Math.round(cfg.step_dt / model.opt.timestep);
    if (hinges.length !== cfg.joint_names.length) throw new Error('joint count differs from the policy');
  }
  compile();

  // Policy joint order is the robot's hinge joints in model order (checked against the
  // policy's joint_names by tools/sim_check.py); actuators map onto joints by id.
  const N = cfg.joint_names.length, def = cfg.default_joint_pos, scale = cfg.action_scale, R = cfg.ranges;

  const obs = new Float32Array(cfg.obs_dim);
  const last = new Float32Array(N);
  let phase = 0, nextBall = 0;
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

  // v' = R(q)^T v for a unit quaternion q = (w, x, y, z)
  function rotInv(q, v, out, o = 0) {
    const [w, x, y, z] = q, [vx, vy, vz] = v;
    const tx = 2 * (-y * vz + z * vy), ty = 2 * (-z * vx + x * vz), tz = 2 * (-x * vy + y * vx);
    out[o] = vx + w * tx + (-y * tz + z * ty);
    out[o + 1] = vy + w * ty + (-z * tx + x * tz);
    out[o + 2] = vz + w * tz + (-x * ty + y * tx);
  }
  function parkBalls() {
    const qp = data.qpos, qv = data.qvel;
    ballQ.forEach((a, i) => { qp.set([100 + i, 100, BALL_R, 1, 0, 0, 0], a); qv.fill(0, ballV[i], ballV[i] + 6); });
  }

  const sim = {
    get model() { return model; }, get data() { return data; }, mujoco, policy, cfg,
    get robotBodies() { return robotBodies; },
    // Recompile with extra <worldbody> geometry (room obstacles); robot and balls keep their state.
    rebuild(extraWorld) {
      const keep = Float64Array.from(data.qpos);
      compile(extraWorld);
      data.qpos.set(keep); mujoco.mj_forward(model, data);
    },
    cmd: [0, 0, 0],
    fallen: false,
    push: null, // { body, force: [3] } applied at the body's centre of mass while set
    reset(x = 0, y = 0, yaw = 0, keepBalls = false) {
      const qp = data.qpos, saved = keepBalls ? Float64Array.from(qp) : null;
      mujoco.mj_resetData(model, data);
      qp.set([x, y, 0.8, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 0);
      for (let i = 0; i < N; i++) qp[qadr[i]] = def[i];
      const ctrl = data.ctrl;
      for (let k = 0; k < model.nu; k++) ctrl[k] = def[actFor[k]];
      if (saved) ballQ.forEach(a => qp.set(saved.subarray(a, a + 7), a)); else parkBalls();
      mujoco.mj_forward(model, data);
      last.fill(0); phase = 0; this.fallen = false;
    },
    // Launch the next ball from `pos` with velocity `vel` (MuJoCo frame).
    throwBall(pos, vel) {
      const i = nextBall++ % N_BALLS, a = ballQ[i], v = ballV[i];
      data.qpos.set([pos[0], pos[1], pos[2], 1, 0, 0, 0], a);
      data.qvel.set([vel[0], vel[1], vel[2], 0, 0, 0], v);
      return i;
    },
    // One 50 Hz policy step: observe, act, then `decim` physics steps.
    step() {
      const qp = data.qpos, qv = data.qvel;
      const q = [qp[3], qp[4], qp[5], qp[6]];
      // Yaw rate goes past the deploy range (±1): tools/sim_check.py shows a stable 1.6 rad/s at a 2.0 command.
      const c = [clamp(this.cmd[0], R.lin_vel_x), clamp(this.cmd[1], R.lin_vel_y), clamp(this.cmd[2], [-2, 2])];
      obs[0] = qv[3]; obs[1] = qv[4]; obs[2] = qv[5];       // free-joint angular velocity is body-frame
      rotInv(q, [0, 0, -1], obs, 3);                          // projected gravity
      obs[6] = c[0]; obs[7] = c[1]; obs[8] = c[2];
      phase = (phase + cfg.step_dt / cfg.gait_period) % 1;
      const moving = Math.hypot(c[0], c[1], c[2]) >= 0.1;
      obs[9] = moving ? Math.sin(2 * Math.PI * phase) : 0;
      obs[10] = moving ? Math.cos(2 * Math.PI * phase) : 0;
      for (let i = 0; i < N; i++) {
        obs[11 + i] = qp[qadr[i]] - def[i];
        obs[11 + N + i] = qv[vadr[i]];
        obs[11 + 2 * N + i] = last[i];
      }
      const a = policy.act(obs);
      last.set(a);
      const ctrl = data.ctrl;
      for (let k = 0; k < model.nu; k++) { const i = actFor[k]; ctrl[k] = def[i] + scale[i] * a[i]; }
      const xf = data.xfrc_applied;
      xf.fill(0);
      if (this.push) xf.set(this.push.force, this.push.body * 6);
      for (let s = 0; s < decim; s++) mujoco.mj_step(model, data);
      if (data.qpos[2] < 0.45) this.fallen = true;
    },
    bodyVel() {
      const qp = data.qpos, qv = data.qvel, out = [0, 0, 0];
      rotInv([qp[3], qp[4], qp[5], qp[6]], [qv[0], qv[1], qv[2]], out);
      return [out[0], out[1], qv[5]];
    },
  };
  sim.reset();
  return sim;
}
