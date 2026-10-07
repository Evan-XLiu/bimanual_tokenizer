"use strict";
const byId = (id) => document.getElementById(id);
const tokenInput = byId("token");
tokenInput.value = new URLSearchParams(location.hash.slice(1)).get("token") || "";
let xrSession = null, reference = null, episode = null, seq = 0, epoch = 0;
let pending = [], sending = null, failed = false, ending = false, timer = null;
let gl, layer, program, locations;

async function api(path, data) {
  const response = await fetch(`/api/${path}`, {
    method: data === undefined ? "GET" : "POST",
    headers: {"Authorization": `Bearer ${tokenInput.value.trim()}`, "Content-Type": "application/json"},
    body: data === undefined ? undefined : JSON.stringify(data),
    signal: AbortSignal.timeout(1500),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function enqueue(row) {
  if (failed || !episode) return;
  if (pending.length >= 240) {
    fail(new Error("发送队列已满；本次轨迹中断，请检查网络后重新进入 XR。"));
    return;
  }
  pending.push({...row, seq: seq++, client_time_ms: performance.now()});
}

function event(name, details = {}) { enqueue({type: "event", name, details}); }

function fail(error) {
  if (failed) return;
  failed = true;
  byId("message").textContent = `采集中断：${error.message}。服务器会标记超时；不要将此轨迹作为完整演示。`;
  if (xrSession) xrSession.end().catch(() => {});
}

async function flush() {
  if (sending) return sending;
  if (!pending.length || failed || !episode) return;
  const records = pending.splice(0);
  sending = api("batch", {episode_id: episode, records}).then((result) => {
    if (result.ack !== records[records.length - 1].seq) throw new Error("序号确认失败");
  }).catch(fail).finally(() => { sending = null; });
  return sending;
}

function pose(value) {
  if (!value) return null;
  const p = value.transform.position, q = value.transform.orientation;
  return {position: [p.x, p.y, p.z], orientation_xyzw: [q.x, q.y, q.z, q.w],
    emulated_position: value.emulatedPosition};
}

function controller(frame, source) {
  const gamepad = source.gamepad;
  return {profiles: Array.from(source.profiles), target_ray_mode: source.targetRayMode,
    grip: source.gripSpace ? pose(frame.getPose(source.gripSpace, reference)) : null,
    target_ray: pose(frame.getPose(source.targetRaySpace, reference)),
    mapping: gamepad?.mapping || null, gamepad_timestamp_ms: gamepad?.timestamp ?? null,
    buttons: gamepad ? Array.from(gamepad.buttons, b => ({value: b.value, pressed: b.pressed, touched: b.touched})) : [],
    axes: gamepad ? Array.from(gamepad.axes) : []};
}

function initRenderer() {
  const canvas = document.createElement("canvas");
  gl = canvas.getContext("webgl", {xrCompatible: true, alpha: true});
  if (!gl) throw new Error("WebGL unavailable");
  function shader(type, code) {
    const item = gl.createShader(type);
    gl.shaderSource(item, code); gl.compileShader(item);
    if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
    return item;
  }
  program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER,
    "uniform mat4 projection; uniform mat4 view; uniform vec3 position; void main(){gl_Position=projection*view*vec4(position,1.0);gl_PointSize=18.0;}"));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER,
    "precision mediump float; uniform vec3 color; void main(){gl_FragColor=vec4(color,1.0);}"));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  locations = Object.fromEntries(["projection", "view", "position", "color"].map(k => [k, gl.getUniformLocation(program, k)]));
}

function onFrame(time, frame) {
  if (ending || failed || !xrSession) return;
  xrSession.requestAnimationFrame(onFrame);
  try {
    const viewer = frame.getViewerPose(reference);
    const row = {type: "frame", xr_frame_time_ms: time, reference_epoch: epoch,
      visibility: xrSession.visibilityState, head: pose(viewer), left: null, right: null};
    for (const source of xrSession.inputSources) {
      if (source.handedness === "left" || source.handedness === "right") {
        row[source.handedness] = controller(frame, source);
      }
    }
    enqueue(row);
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0.025, 0.035, 0.05, xrSession.environmentBlendMode === "opaque" ? 1 : 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (viewer) {
      gl.useProgram(program);
      for (const view of viewer.views) {
        const v = layer.getViewport(view);
        gl.viewport(v.x, v.y, v.width, v.height);
        gl.uniformMatrix4fv(locations.projection, false, view.projectionMatrix);
        gl.uniformMatrix4fv(locations.view, false, view.transform.inverse.matrix);
        for (const [side, color] of [["left", [0.1, 0.8, 1]], ["right", [1, 0.6, 0.1]]]) {
          if (row[side]?.grip) {
            gl.uniform3fv(locations.position, row[side].grip.position);
            gl.uniform3fv(locations.color, color);
            gl.drawArrays(gl.POINTS, 0, 1);
          }
        }
      }
    }
  } catch (error) { fail(error); }
}

async function finish() {
  ending = true;
  clearInterval(timer);
  event("session_end");
  await flush();
  // A batch might have been in flight when the final event was queued.
  await flush();
  try {
    if (episode && !failed) {
      const summary = await api("end", {episode_id: episode});
      byId("message").textContent = `已保存 ${summary.frames} 帧；双手有效 ${summary.valid_both_frames} 帧。`;
    }
  } catch (error) { fail(error); }
  xrSession = null; episode = null; pending = [];
  byId("stop").disabled = true;
  byId("start").disabled = !navigator.xr;
  if (gl) gl.getExtension("WEBGL_lose_context")?.loseContext();
}

byId("start").onclick = async () => {
  byId("start").disabled = true;
  failed = false; ending = false; seq = 0; epoch = 0; pending = [];
  try {
    if (!tokenInput.value.trim()) throw new Error("请填写电脑终端显示的连接码");
    xrSession = await navigator.xr.requestSession(byId("mode").value, {
      requiredFeatures: ["local-floor"], optionalFeatures: ["dom-overlay"],
      domOverlay: {root: byId("overlay")},
    });
    xrSession.addEventListener("end", finish, {once: true});
    initRenderer();
    await gl.makeXRCompatible();
    layer = new XRWebGLLayer(xrSession, gl);
    xrSession.updateRenderState({baseLayer: layer});
    reference = await xrSession.requestReferenceSpace("local-floor");
    const metadata = {reference_space: "local-floor", performance_time_origin_ms: performance.timeOrigin,
      user_agent: navigator.userAgent, mode: byId("mode").value,
      environment_blend_mode: xrSession.environmentBlendMode, device_sensor_timestamp: "unavailable"};
    episode = (await api("start", metadata)).episode_id;
    reference.addEventListener("reset", (e) => {
      epoch++;
      event("reference_reset", {reference_epoch: epoch, transform_matrix: e.transform ? Array.from(e.transform.matrix) : null});
    });
    for (const name of ["visibilitychange", "inputsourceschange", "selectstart", "selectend", "squeezestart", "squeezeend"]) {
      xrSession.addEventListener(name, (e) => event(name, {
        visibility: xrSession.visibilityState, handedness: e.inputSource?.handedness ?? null,
        added: e.added ? Array.from(e.added, s => ({handedness: s.handedness, profiles: Array.from(s.profiles)})) : [],
        removed: e.removed ? Array.from(e.removed, s => ({handedness: s.handedness, profiles: Array.from(s.profiles)})) : [],
      }));
    }
    event("session_start", metadata);
    timer = setInterval(flush, 100);
    byId("stop").disabled = false;
    byId("message").textContent = "正在采集。左手蓝点、右手橙点；退出 XR 即保存。";
    xrSession.requestAnimationFrame(onFrame);
  } catch (error) {
    fail(error);
    if (!xrSession) byId("start").disabled = false;
  }
};
byId("stop").onclick = () => xrSession?.end();
byId("overlay").addEventListener("beforexrselect", e => e.preventDefault());

function describe(ctrl) {
  if (!ctrl) return "未连接";
  const location = ctrl.grip ? ctrl.grip.position.map(v => v.toFixed(3)).join(", ") : "追踪丢失";
  return `${location} | buttons=${ctrl.buttons.map(b => b.value.toFixed(2)).join(",")} | ${ctrl.mapping}`;
}
setInterval(async () => {
  if (!tokenInput.value.trim()) return;
  try {
    const state = await api("status");
    byId("status").textContent = state.active ?
      `接收帧数 ${state.frames} | 最近接收 ${state.receive_age_s.toFixed(2)} 秒前\n左：${describe(state.latest?.left)}\n右：${describe(state.latest?.right)}` :
      `等待 XR 连接\n${state.last_summary ? JSON.stringify(state.last_summary, null, 2) : ""}`;
  } catch (error) { byId("status").textContent = `服务器不可达：${error.message}`; }
}, 1000);

if (!isSecureContext || !navigator.xr) {
  byId("message").textContent = "当前浏览器不能开始 WebXR。请用 Quest 浏览器访问受信任的 HTTPS（或 USB 转发的 localhost）。电脑页面可监看状态。";
} else {
  byId("start").disabled = false;
  byId("message").textContent = "准备就绪。拿起两个手柄，点击进入 XR；若透视 AR 不支持，可选择沉浸 VR。";
}
