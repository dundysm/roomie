// MuJoCo + the unitree_rl_mjlab G1 velocity policy, ported 1:1 from tools/sim_check.py.
import loadMujoco from '@mujoco/mujoco';

const HINGE = 3; // mjJNT_HINGE

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
  const [xml, cfg, bin] = await Promise.all([
    fetch(base + 'g1.xml').then(r => r.text()),
    fetch(base + 'policy.json').then(r => r.json()),
    fetch(base + 'policy.bin').then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
  ]);
  const FS = mujoco.FS;
  FS.mkdir('/g1'); FS.mkdir('/g1/assets');
  FS.writeFile('/g1/g1.xml', xml);
  const meshFiles = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map(m => m[1]))];
  await Promise.all(meshFiles.map(async f => FS.writeFile('/g1/assets/' + f, await fetchBytes(base + 'assets/' + f))));

  const policy = new Policy(cfg, bin);
  const baseScene = new TextDecoder().decode(await fetchBytes(base + 'scene.xml'));
  let model, data, hinges, qadr, vadr, actFor, decim;
  function compile(extraWorld = '') {
    FS.writeFile('/g1/scene.xml', baseScene.replace('</worldbody>', extraWorld + '\n  </worldbody>'));
    const m = mujoco.MjModel.from_xml_path('/g1/scene.xml');
    if (data) data.delete();
    if (model) model.delete();
    model = m; data = new mujoco.MjData(model);
    hinges = [];
    for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === HINGE) hinges.push(j);
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
  let phase = 0;
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

  // v' = R(q)^T v for a unit quaternion q = (w, x, y, z)
  function rotInv(q, v, out, o = 0) {
    const [w, x, y, z] = q, [vx, vy, vz] = v;
    const tx = 2 * (-y * vz + z * vy), ty = 2 * (-z * vx + x * vz), tz = 2 * (-x * vy + y * vx);
    out[o] = vx + w * tx + (-y * tz + z * ty);
    out[o + 1] = vy + w * ty + (-z * tx + x * tz);
    out[o + 2] = vz + w * tz + (-x * ty + y * tx);
  }

  const sim = {
    get model() { return model; }, get data() { return data; }, mujoco, policy, cfg,
    // Recompile with extra <worldbody> geometry (room obstacles); the robot keeps its pose.
    rebuild(extraWorld) {
      const keep = Float64Array.from(data.qpos);
      compile(extraWorld);
      data.qpos.set(keep); mujoco.mj_forward(model, data);
    },
    cmd: [0, 0, 0],
    fallen: false,
    reset(x = 0, y = 0, yaw = 0) {
      mujoco.mj_resetDataKeyframe(model, data, 0);
      const qp = data.qpos;
      qp[0] = x; qp[1] = y;
      qp[3] = Math.cos(yaw / 2); qp[4] = 0; qp[5] = 0; qp[6] = Math.sin(yaw / 2);
      mujoco.mj_forward(model, data);
      last.fill(0); phase = 0; this.fallen = false;
    },
    // One 50 Hz policy step: observe, act, then `decim` physics steps.
    step() {
      const qp = data.qpos, qv = data.qvel;
      const q = [qp[3], qp[4], qp[5], qp[6]];
      const c = [clamp(this.cmd[0], R.lin_vel_x), clamp(this.cmd[1], R.lin_vel_y), clamp(this.cmd[2], R.ang_vel_z)];
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
