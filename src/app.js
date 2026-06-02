import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

const canvas = document.querySelector("#gameCanvas");
const ctx = canvas.getContext("2d");
const video = document.querySelector("#cameraVideo");
const debugCanvas = document.querySelector("#debugCanvas");
const debugCtx = debugCanvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const startOverlay = document.querySelector("#startOverlay");
const statusText = document.querySelector("#statusText");
const debugToggle = document.querySelector("#debugToggle");
const cameraPanel = document.querySelector("#cameraPanel");
const messageLayer = document.querySelector("#messageLayer");

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";

const palette = ["#ffcf4f", "#ff7f6e", "#5ee6a8", "#b79cff"];
const hitMessages = ["泡泡弹打中啦！", "海路清出来啦！", "泡泡散开啦！", "障碍变泡泡啦！"];
const bumpMessages = ["潜艇晃了一下！", "冒泡泡啦！", "轻轻碰到了！"];
const SHOOT_SPEED_THRESHOLD = 720;
const SHOOT_COOLDOWN_MS = 560;
const PROJECTILE_SPEED = 520;
const POSE_INTERVAL_MS = 33;

const state = {
  width: 1280,
  height: 720,
  dpr: 1,
  running: false,
  debug: false,
  poseLandmarker: null,
  lastVideoTime: -1,
  lastPoseAt: 0,
  lastFrameTime: performance.now(),
  players: [],
  obstacles: [],
  projectiles: [],
  particles: [],
  bubbles: [],
  seaPlants: [],
  ripples: [],
  nextObstacleAt: 0,
  lastMessageAt: 0,
  cameraReady: false,
  poseReady: false,
  demoMode: false,
  loopStarted: false,
  pointer: {
    x: window.innerWidth * 0.5,
    y: window.innerHeight * 0.5,
    active: false,
    pressed: false,
  },
};

window.submarineMvpDebug = {
  state,
};

const landmarkNames = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftWrist: 15,
  rightWrist: 16,
};

function resizeCanvas() {
  state.dpr = 1;
  state.width = window.innerWidth;
  state.height = window.innerHeight;
  canvas.width = Math.floor(state.width * state.dpr);
  canvas.height = Math.floor(state.height * state.dpr);
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);

  debugCanvas.width = 640;
  debugCanvas.height = 480;
  seedScenery();
}

function seedScenery() {
  state.seaPlants = Array.from({ length: 14 }, (_, index) => ({
    x: (index / 13) * state.width + random(-24, 24),
    h: random(42, 110),
    sway: random(0, Math.PI * 2),
    color: index % 2 ? "#27b58f" : "#34a6a0",
  }));

  if (state.bubbles.length === 0) {
    state.bubbles = Array.from({ length: 30 }, () => ({
      x: random(0, state.width),
      y: random(0, state.height),
      r: random(3, 10),
      speed: random(12, 34),
      wobble: random(0, Math.PI * 2),
    }));
  }
}

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(from, to, amount) {
  return from + (to - from) * amount;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function normalize(vector, fallback = { x: 1, y: 0 }) {
  const length = Math.hypot(vector.x, vector.y);
  if (length < 0.001) return { ...fallback, length: 0 };
  return { x: vector.x / length, y: vector.y / length, length };
}

function mirroredX(normalizedX) {
  return (1 - normalizedX) * state.width;
}

function toScreenPoint(landmark) {
  return {
    x: mirroredX(landmark.x),
    y: landmark.y * state.height,
    visible: landmark.visibility ?? landmark.presence ?? 1,
  };
}

function goodLandmark(landmark, threshold = 0.34) {
  if (!landmark) return false;
  const confidence = landmark.visibility ?? landmark.presence ?? 1;
  return confidence >= threshold;
}

function averagePoints(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return {
    x: valid.reduce((sum, point) => sum + point.x, 0) / valid.length,
    y: valid.reduce((sum, point) => sum + point.y, 0) / valid.length,
    visible: valid.reduce((sum, point) => sum + point.visible, 0) / valid.length,
  };
}

async function startGame() {
  startButton.disabled = true;
  setStatus("正在请求摄像头...");

  try {
    await startCamera();
    setStatus("正在加载姿态识别模型...");
    await loadPoseModel();
    state.demoMode = false;
    startOverlay.classList.add("is-hidden");
    setStatus("看着半透明摄像头背景，移动脑袋躲避障碍，挥动左右手发射泡泡弹。");
    beginLoop();
  } catch (error) {
    enterDemoMode(error);
  }
}

function beginLoop() {
  state.running = true;
  state.lastFrameTime = performance.now();
  if (!state.loopStarted) {
    state.loopStarted = true;
    requestAnimationFrame(loop);
  }
}

function enterDemoMode(error) {
  state.demoMode = true;
  if (!state.cameraReady) {
    video.classList.remove("is-visible");
  }
  startOverlay.classList.add("is-hidden");
  const reason = formatStartupError(error);
  setStatus(`${reason} 已进入鼠标演示模式。移动鼠标躲避，点击发射泡泡弹。`);
  beginLoop();
}

function formatStartupError(error) {
  const name = error?.name || "";
  const message = error?.message || "";
  if (name === "NotAllowedError" || /permission/i.test(message)) {
    return "摄像头权限被拒绝。";
  }
  if (name === "NotFoundError" || /requested device not found/i.test(message)) {
    return "没有找到可用摄像头。";
  }
  if (/network|fetch|load/i.test(message)) {
    return "姿态识别模型暂时无法加载。";
  }
  return "体感启动暂时不可用。";
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("当前浏览器不支持摄像头访问。");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
  state.cameraReady = true;
  video.classList.add("is-visible");
}

async function loadPoseModel() {
  if (state.poseLandmarker) return;

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.42,
    minPosePresenceConfidence: 0.42,
    minTrackingConfidence: 0.42,
  });
  state.poseReady = true;
}

function loop(now) {
  const dt = Math.min((now - state.lastFrameTime) / 1000, 0.05);
  state.lastFrameTime = now;

  updatePose(now);
  updateWorld(dt, now);
  drawWorld(now);

  if (state.running) {
    requestAnimationFrame(loop);
  }
}

function updatePose(now) {
  if (state.demoMode) {
    return;
  }

  if (!state.poseLandmarker || !video.videoWidth || video.currentTime === state.lastVideoTime) {
    return;
  }

  if (now - state.lastPoseAt < POSE_INTERVAL_MS) {
    return;
  }

  const result = state.poseLandmarker.detectForVideo(video, now);
  state.lastPoseAt = now;
  state.lastVideoTime = video.currentTime;
  updatePlayers(result.landmarks || [], now);
  if (state.debug) {
    drawDebug(result.landmarks || []);
  }
}

function updatePlayers(poses, now) {
  const mapped = poses
    .map((landmarks, poseIndex) => mapPoseToPlayer(landmarks, poseIndex))
    .filter(Boolean)
    .sort((a, b) => a.head.x - b.head.x)
    .slice(0, 4);

  mapped.forEach((pose, index) => {
    const existing = state.players[index] || createPlayer(index, pose);
    const smooth = existing.active ? 0.22 : 1;
    existing.active = true;
    existing.lastSeenAt = now;
    existing.color = palette[index % palette.length];
    existing.head.x = lerp(existing.head.x, pose.head.x, smooth);
    existing.head.y = lerp(existing.head.y, pose.head.y, smooth);
    existing.sub.x = lerp(existing.sub.x, pose.targetSub.x, smooth);
    existing.sub.y = lerp(existing.sub.y, pose.targetSub.y, smooth);
    existing.shoulder = pose.shoulder;
    updateArmFromPose(existing, "left", pose.leftHand, pose.leftArmTarget, now, smooth);
    updateArmFromPose(existing, "right", pose.rightHand, pose.rightArmTarget, now, smooth);
    existing.landmarks = pose.landmarks;
    state.players[index] = existing;
  });

  for (let index = mapped.length; index < state.players.length; index += 1) {
    const player = state.players[index];
    if (now - player.lastSeenAt > 1200) {
      player.active = false;
    }
  }
}

function createPlayer(index, pose) {
  return {
    id: index + 1,
    color: palette[index % palette.length],
    active: false,
    lastSeenAt: performance.now(),
    head: { ...pose.head },
    shoulder: { ...pose.shoulder },
    sub: { ...pose.targetSub },
    arms: {
      left: createArm("left", pose.leftHand || pose.fallbackLeftHand, pose.leftArmTarget),
      right: createArm("right", pose.rightHand || pose.fallbackRightHand, pose.rightArmTarget),
    },
    hurtUntil: 0,
    hurtPhase: 0,
    landmarks: pose.landmarks,
  };
}

function createArm(side, hand, target) {
  return {
    side,
    hand: { ...hand },
    prevHand: { ...hand },
    end: { ...target },
    direction: normalize({ x: target.x, y: target.y }, side === "left" ? { x: -1, y: 0 } : { x: 1, y: 0 }),
    speed: 0,
    lastHandAt: performance.now(),
    cooldownUntil: 0,
    flashUntil: 0,
  };
}

function updateArmFromPose(player, side, hand, target, now, smooth) {
  if (!hand || !target) {
    return;
  }

  const arm = player.arms[side];
  const elapsed = Math.max((now - arm.lastHandAt) / 1000, 0.016);
  const speed = distance(hand, arm.hand) / elapsed;
  arm.prevHand = { ...arm.hand };
  arm.hand = { ...hand };
  arm.lastHandAt = now;
  arm.speed = speed;
  arm.end.x = lerp(arm.end.x, target.x, smooth);
  arm.end.y = lerp(arm.end.y, target.y, smooth);
  arm.direction = normalize({ x: arm.end.x - player.sub.x, y: arm.end.y - player.sub.y }, arm.direction);

  if (speed > SHOOT_SPEED_THRESHOLD && now > arm.cooldownUntil) {
    fireProjectile(player, side, now);
  }
}

function mapPoseToPlayer(landmarks, poseIndex) {
  const head = averagePoints(
    [
      landmarkNames.nose,
      landmarkNames.leftEye,
      landmarkNames.rightEye,
      landmarkNames.leftEar,
      landmarkNames.rightEar,
    ].map((index) => (goodLandmark(landmarks[index]) ? toScreenPoint(landmarks[index]) : null)),
  );
  const leftShoulder = goodLandmark(landmarks[landmarkNames.leftShoulder])
    ? toScreenPoint(landmarks[landmarkNames.leftShoulder])
    : null;
  const rightShoulder = goodLandmark(landmarks[landmarkNames.rightShoulder])
    ? toScreenPoint(landmarks[landmarkNames.rightShoulder])
    : null;
  const leftHand = goodLandmark(landmarks[landmarkNames.leftWrist])
    ? toScreenPoint(landmarks[landmarkNames.leftWrist])
    : null;
  const rightHand = goodLandmark(landmarks[landmarkNames.rightWrist])
    ? toScreenPoint(landmarks[landmarkNames.rightWrist])
    : null;
  const shoulder =
    averagePoints([leftShoulder, rightShoulder]) || {
      x: head?.x ?? state.width * 0.5,
      y: (head?.y ?? state.height * 0.45) + 92,
      visible: 0.5,
    };

  if (!head) return null;

  const playerBandOffset = (poseIndex - 0.5) * 4;
  const targetSub = {
    x: clamp(head.x + playerBandOffset, 72, state.width - 72),
    y: clamp(head.y + 56, 106, state.height - 96),
  };
  const fallbackLeftHand = {
    x: targetSub.x - 118,
    y: targetSub.y + 8,
    visible: 0.5,
  };
  const fallbackRightHand = {
    x: targetSub.x + 118,
    y: targetSub.y + 8,
    visible: 0.5,
  };
  const leftArmTarget = leftHand ? mapArmTarget(targetSub, shoulder, leftHand, "left") : mapArmTarget(targetSub, shoulder, fallbackLeftHand, "left");
  const rightArmTarget = rightHand ? mapArmTarget(targetSub, shoulder, rightHand, "right") : mapArmTarget(targetSub, shoulder, fallbackRightHand, "right");

  return {
    head,
    shoulder,
    leftHand,
    rightHand,
    fallbackLeftHand,
    fallbackRightHand,
    targetSub,
    leftArmTarget,
    rightArmTarget,
    landmarks,
  };
}

function mapArmTarget(sub, shoulder, hand, side) {
  const fallback = side === "left" ? { x: -1, y: 0 } : { x: 1, y: 0 };
  const offset = {
    x: (hand.x - shoulder.x) * 1.12,
    y: (hand.y - shoulder.y) * 1.12,
  };
  const normal = normalize(offset, fallback);
  const length = clamp(normal.length, 44, 188);
  return {
    x: clamp(sub.x + normal.x * length, 28, state.width - 28),
    y: clamp(sub.y + normal.y * length, 58, state.height - 58),
  };
}

function updateWorld(dt, now) {
  if (state.demoMode) {
    updateDemoPlayer(dt, now);
  }

  if (state.obstacles.length < 10 && now > state.nextObstacleAt) {
    spawnObstacle();
    state.nextObstacleAt = now + random(780, 1350);
  }

  state.bubbles.forEach((bubble) => {
    bubble.y -= bubble.speed * dt;
    bubble.x += Math.sin(now / 800 + bubble.wobble) * 10 * dt;
    if (bubble.y + bubble.r < 0) {
      bubble.y = state.height + random(10, 90);
      bubble.x = random(0, state.width);
    }
  });

  state.obstacles.forEach((obstacle) => {
    obstacle.x += obstacle.vx * dt;
    obstacle.y += obstacle.vy * dt + Math.sin(now / 520 + obstacle.phase) * 10 * dt;
    obstacle.rotation += obstacle.spin * dt;
    if (obstacle.x < -90 || obstacle.x > state.width + 90 || obstacle.y < -90 || obstacle.y > state.height + 90) {
      obstacle.remove = true;
    }
  });

  state.projectiles.forEach((projectile) => {
    projectile.x += projectile.vx * dt;
    projectile.y += projectile.vy * dt;
    projectile.life -= dt;
    projectile.pulse += dt * 8;
    if (
      projectile.life <= 0 ||
      projectile.x < -40 ||
      projectile.x > state.width + 40 ||
      projectile.y < -40 ||
      projectile.y > state.height + 40
    ) {
      projectile.remove = true;
    }
  });

  state.particles.forEach((particle) => {
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    particle.vy += 18 * dt;
    particle.life -= dt;
    particle.rotation += particle.spin * dt;
  });

  state.ripples.forEach((ripple) => {
    ripple.life -= dt;
    ripple.r += 95 * dt;
  });

  handleCollisions(now);

  state.obstacles = state.obstacles.filter((obstacle) => !obstacle.remove);
  state.projectiles = state.projectiles.filter((projectile) => !projectile.remove);
  state.particles = state.particles.filter((particle) => particle.life > 0);
  state.ripples = state.ripples.filter((ripple) => ripple.life > 0);
}

function updateDemoPlayer(dt, now) {
  const targetSub = {
    x: clamp(state.pointer.x, 72, state.width - 72),
    y: clamp(state.pointer.y, 106, state.height - 96),
  };
  const player =
    state.players[0] ||
    createPlayer(0, {
      head: { x: targetSub.x, y: targetSub.y - 56, visible: 1 },
      shoulder: { x: targetSub.x, y: targetSub.y, visible: 1 },
      leftHand: { x: targetSub.x - 128, y: targetSub.y + 20, visible: 1 },
      rightHand: { x: targetSub.x + 128, y: targetSub.y + 20, visible: 1 },
      targetSub,
      leftArmTarget: { x: targetSub.x - 128, y: targetSub.y + 20 },
      rightArmTarget: { x: targetSub.x + 128, y: targetSub.y + 20 },
      landmarks: [],
    });

  const smooth = clamp(dt * 8, 0, 1);
  const leftTarget = {
    x: clamp(targetSub.x - 118 + Math.sin(now / 260) * 46, 28, state.width - 28),
    y: clamp(targetSub.y + Math.cos(now / 310) * 78, 58, state.height - 58),
  };
  const rightTarget = {
    x: clamp(targetSub.x + 118 + Math.cos(now / 260) * 46, 28, state.width - 28),
    y: clamp(targetSub.y + Math.sin(now / 300) * 78, 58, state.height - 58),
  };

  player.active = true;
  player.lastSeenAt = now;
  player.color = palette[0];
  player.head = { x: targetSub.x, y: targetSub.y - 56, visible: 1 };
  player.shoulder = { x: targetSub.x, y: targetSub.y, visible: 1 };
  player.sub.x = lerp(player.sub.x, targetSub.x, smooth);
  player.sub.y = lerp(player.sub.y, targetSub.y, smooth);

  updateDemoArm(player, "left", leftTarget, now, smooth);
  updateDemoArm(player, "right", rightTarget, now, smooth);

  if (state.pointer.pressed && now > player.arms.right.cooldownUntil) {
    fireProjectile(player, "right", now);
  }
  state.players[0] = player;
}

function updateDemoArm(player, side, target, now, smooth) {
  const arm = player.arms[side];
  arm.end.x = lerp(arm.end.x, target.x, smooth);
  arm.end.y = lerp(arm.end.y, target.y, smooth);
  arm.hand = { x: arm.end.x, y: arm.end.y, visible: 1 };
  arm.direction = normalize({ x: arm.end.x - player.sub.x, y: arm.end.y - player.sub.y }, side === "left" ? { x: -1, y: 0 } : { x: 1, y: 0 });
  arm.speed = 0;
  if (now > arm.cooldownUntil && Math.sin(now / 340 + (side === "left" ? 0 : Math.PI)) > 0.96) {
    fireProjectile(player, side, now);
  }
}

function fireProjectile(player, side, now) {
  const arm = player.arms[side];
  const direction = normalize({ x: arm.end.x - player.sub.x, y: arm.end.y - player.sub.y }, arm.direction);
  const start = {
    x: arm.end.x + direction.x * 12,
    y: arm.end.y + direction.y * 12,
  };
  state.projectiles.push({
    ownerId: player.id,
    color: player.color,
    x: start.x,
    y: start.y,
    vx: direction.x * PROJECTILE_SPEED,
    vy: direction.y * PROJECTILE_SPEED,
    r: 12,
    life: 1.45,
    pulse: 0,
  });
  arm.cooldownUntil = now + SHOOT_COOLDOWN_MS;
  arm.flashUntil = now + 180;
  burst(start.x, start.y, player.color, "tinyBubble", 6);
}

function spawnObstacle() {
  const side = Math.floor(random(0, 4));
  const types = ["rock", "urchin", "crate", "bubbleRock"];
  let x = 0;
  let y = 0;
  let vx = 0;
  let vy = 0;

  if (side === 0) {
    x = -48;
    y = random(100, state.height - 96);
    vx = random(42, 92);
    vy = random(-18, 18);
  } else if (side === 1) {
    x = state.width + 48;
    y = random(100, state.height - 96);
    vx = -random(42, 92);
    vy = random(-18, 18);
  } else if (side === 2) {
    x = random(96, state.width - 96);
    y = -48;
    vx = random(-28, 28);
    vy = random(36, 78);
  } else {
    x = random(96, state.width - 96);
    y = state.height + 48;
    vx = random(-28, 28);
    vy = -random(30, 64);
  }

  state.obstacles.push({
    type: types[Math.floor(random(0, types.length))],
    x,
    y,
    vx,
    vy,
    r: random(22, 34),
    phase: random(0, Math.PI * 2),
    rotation: random(-0.5, 0.5),
    spin: random(-0.8, 0.8),
  });
}

function handleCollisions(now) {
  const activePlayers = state.players.filter((player) => player.active);

  state.projectiles.forEach((projectile) => {
    state.obstacles.forEach((obstacle) => {
      if (!projectile.remove && !obstacle.remove && distance(projectile, obstacle) < projectile.r + obstacle.r) {
        projectile.remove = true;
        obstacle.remove = true;
        burst(obstacle.x, obstacle.y, projectile.color, obstacle.type === "urchin" ? "star" : "bubble", 22);
        state.ripples.push({ x: obstacle.x, y: obstacle.y, r: 8, life: 0.75, color: projectile.color });
        showMessage(hitMessages[Math.floor(random(0, hitMessages.length))]);
      }
    });
  });

  activePlayers.forEach((player) => {
    state.obstacles.forEach((obstacle) => {
      if (!obstacle.remove && distance(player.sub, obstacle) < obstacle.r + 44) {
        obstacle.remove = true;
        player.hurtUntil = now + 520;
        player.hurtPhase = random(0, Math.PI * 2);
        burst(obstacle.x, obstacle.y, "#ffffff", "bubble", 18);
        state.ripples.push({ x: player.sub.x, y: player.sub.y, r: 10, life: 0.65, color: player.color });
        showMessage(bumpMessages[Math.floor(random(0, bumpMessages.length))]);
      }
    });
  });
}

function burst(x, y, color, shape, count = 18) {
  for (let index = 0; index < count; index += 1) {
    const angle = random(0, Math.PI * 2);
    const speed = random(54, 150);
    state.particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - random(16, 50),
      r: random(3, shape === "tinyBubble" ? 6 : 9),
      life: random(0.55, 1.05),
      maxLife: 1.05,
      color: index % 3 === 0 ? "#ffffff" : color,
      shape,
      rotation: random(0, Math.PI * 2),
      spin: random(-5, 5),
    });
  }
}

function showMessage(text) {
  const now = performance.now();
  if (now - state.lastMessageAt < 520) return;
  state.lastMessageAt = now;

  const message = document.createElement("div");
  message.className = "joy-message";
  message.textContent = text;
  messageLayer.append(message);
  window.setTimeout(() => message.remove(), 1450);
}

function drawWorld(now) {
  ctx.clearRect(0, 0, state.width, state.height);
  drawOcean(now);
  drawObstacles();
  drawProjectiles(now);
  drawParticles();
  drawPlayers(now);
  drawForeground(now);
}

function drawOcean(now) {
  const gradient = ctx.createLinearGradient(0, 0, 0, state.height);
  gradient.addColorStop(0, "#82dcff");
  gradient.addColorStop(0.5, "#35aede");
  gradient.addColorStop(1, "#1c82c2");
  ctx.save();
  ctx.globalAlpha = state.cameraReady && !state.demoMode ? 0.3 : 1;
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, state.width, state.height);
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = state.cameraReady && !state.demoMode ? 0.1 : 0.2;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  for (let row = 0; row < 7; row += 1) {
    const y = 80 + row * 82;
    ctx.beginPath();
    for (let x = -30; x < state.width + 32; x += 24) {
      const waveY = y + Math.sin(x / 60 + now / 1300 + row) * 10;
      if (x === -30) ctx.moveTo(x, waveY);
      else ctx.lineTo(x, waveY);
    }
    ctx.stroke();
  }
  ctx.restore();

  state.bubbles.forEach((bubble) => {
    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(bubble.x, bubble.y, bubble.r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
}

function drawForeground(now) {
  ctx.save();
  ctx.globalAlpha = 0.85;
  state.seaPlants.forEach((plant) => {
    ctx.strokeStyle = plant.color;
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(plant.x, state.height + 8);
    const midX = plant.x + Math.sin(now / 750 + plant.sway) * 12;
    ctx.quadraticCurveTo(midX, state.height - plant.h * 0.45, plant.x + Math.sin(now / 840 + plant.sway) * 20, state.height - plant.h);
    ctx.stroke();
  });
  ctx.restore();
}

function drawPlayers(now) {
  const activePlayers = state.players.filter((player) => player.active);

  if (!activePlayers.length) {
    drawIdleSubmarine(now);
    return;
  }

  activePlayers.forEach((player) => {
    drawSubmarine(player, now);
    if (state.debug) {
      drawPlayerGuides(player);
    }
  });
}

function drawIdleSubmarine(now) {
  const sub = {
    x: state.width * 0.5,
    y: state.height * 0.55 + Math.sin(now / 600) * 18,
  };
  const player = {
    id: 1,
    color: "#ffcf4f",
    sub,
    arms: {
      left: { end: { x: sub.x - 120, y: sub.y + Math.sin(now / 500) * 64 }, flashUntil: 0 },
      right: { end: { x: sub.x + 120, y: sub.y + Math.cos(now / 500) * 64 }, flashUntil: 0 },
    },
    hurtUntil: 0,
  };
  drawSubmarine(player, now, true);
}

function drawSubmarine(player, now, idle = false) {
  const hurtActive = now < player.hurtUntil;
  const bob = Math.sin(now / 520 + player.id) * 3;
  const shake = hurtActive ? Math.sin(now / 32 + player.hurtPhase) * 8 : 0;
  const sub = { x: player.sub.x + shake, y: player.sub.y + bob };

  drawArm(sub, player.arms.left, player.color, now, idle);
  drawArm(sub, player.arms.right, player.color, now, idle);

  ctx.save();
  ctx.translate(sub.x, sub.y);
  if (hurtActive) {
    ctx.rotate(Math.sin(now / 48 + player.hurtPhase) * 0.08);
    ctx.globalAlpha = 0.68 + Math.sin(now / 42) * 0.18;
  }
  ctx.fillStyle = player.color;
  roundRect(ctx, -52, -28, 104, 56, 28);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.72)";
  ctx.beginPath();
  ctx.arc(12, -6, 13, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#31546b";
  ctx.beginPath();
  ctx.arc(12, -6, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = player.color;
  roundRect(ctx, -12, -48, 28, 18, 8);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.42)";
  roundRect(ctx, -42, -14, 42, 10, 5);
  ctx.fill();
  ctx.fillStyle = "rgba(25,68,88,0.72)";
  ctx.beginPath();
  ctx.moveTo(-52, 0);
  ctx.lineTo(-78, -20);
  ctx.lineTo(-78, 20);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.font = "900 16px ui-rounded, system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(player.id, -20, 2);
  ctx.restore();
}

function drawArm(sub, arm, color, now, idle) {
  const end = arm.end;
  const activeFlash = now < arm.flashUntil;
  ctx.save();
  ctx.strokeStyle = idle ? "rgba(255,255,255,0.55)" : color;
  ctx.lineWidth = activeFlash ? 13 : 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(sub.x, sub.y);
  const midX = lerp(sub.x, end.x, 0.5);
  const midY = lerp(sub.y, end.y, 0.5) - 12;
  ctx.quadraticCurveTo(midX, midY, end.x, end.y);
  ctx.stroke();
  ctx.fillStyle = activeFlash ? "#ffffff" : color;
  ctx.beginPath();
  ctx.arc(end.x, end.y, activeFlash ? 16 : 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.72)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(end.x, end.y, activeFlash ? 21 : 16, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawProjectiles(now) {
  state.projectiles.forEach((projectile) => {
    ctx.save();
    ctx.translate(projectile.x, projectile.y);
    const pulse = Math.sin(projectile.pulse) * 2;
    ctx.fillStyle = projectile.color;
    ctx.globalAlpha = 0.88;
    ctx.beginPath();
    ctx.arc(0, 0, projectile.r + pulse, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.76;
    ctx.beginPath();
    ctx.arc(0, 0, projectile.r + 5 + pulse, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
}

function drawObstacles() {
  state.obstacles.forEach((obstacle) => {
    ctx.save();
    ctx.translate(obstacle.x, obstacle.y);
    ctx.rotate(obstacle.rotation);
    if (obstacle.type === "rock") drawRock(obstacle.r);
    if (obstacle.type === "urchin") drawUrchin(obstacle.r);
    if (obstacle.type === "crate") drawCrate(obstacle.r);
    if (obstacle.type === "bubbleRock") drawBubbleRock(obstacle.r);
    ctx.restore();
  });
}

function drawRock(r) {
  ctx.fillStyle = "#8ea8b5";
  const points = [
    [0.98, 0],
    [0.68, 0.58],
    [0.22, 0.92],
    [-0.62, 0.76],
    [-0.98, 0.2],
    [-0.78, -0.48],
    [-0.24, -0.92],
    [0.58, -0.72],
  ];
  ctx.beginPath();
  points.forEach(([px, py], index) => {
    const x = px * r;
    const y = py * r;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.26)";
  ctx.beginPath();
  ctx.arc(-r * 0.25, -r * 0.25, r * 0.25, 0, Math.PI * 2);
  ctx.fill();
}

function drawUrchin(r) {
  ctx.strokeStyle = "#6d7c99";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  for (let index = 0; index < 10; index += 1) {
    const angle = (index / 10) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(Math.cos(angle) * r * 0.35, Math.sin(angle) * r * 0.35);
    ctx.lineTo(Math.cos(angle) * r * 1.1, Math.sin(angle) * r * 1.1);
    ctx.stroke();
  }
  ctx.fillStyle = "#9fb0cb";
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.72, 0, Math.PI * 2);
  ctx.fill();
}

function drawCrate(r) {
  ctx.fillStyle = "#d7b46a";
  roundRect(ctx, -r, -r * 0.82, r * 2, r * 1.64, 8);
  ctx.fill();
  ctx.strokeStyle = "rgba(94,72,35,0.45)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(-r * 0.75, -r * 0.54);
  ctx.lineTo(r * 0.75, r * 0.54);
  ctx.moveTo(r * 0.75, -r * 0.54);
  ctx.lineTo(-r * 0.75, r * 0.54);
  ctx.stroke();
}

function drawBubbleRock(r) {
  const gradient = ctx.createRadialGradient(-r * 0.3, -r * 0.35, 2, 0, 0, r);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(1, "#bcecff");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.95, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.82)";
  ctx.lineWidth = 3;
  ctx.stroke();
}

function drawParticles() {
  state.ripples.forEach((ripple) => {
    ctx.save();
    ctx.globalAlpha = clamp(ripple.life / 0.75, 0, 1);
    ctx.strokeStyle = ripple.color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(ripple.x, ripple.y, ripple.r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });

  state.particles.forEach((particle) => {
    const alpha = clamp(particle.life / particle.maxLife, 0, 1);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(particle.x, particle.y);
    ctx.rotate(particle.rotation);
    if (particle.shape === "star") {
      drawStar(0, 0, particle.r, particle.color);
    } else {
      ctx.strokeStyle = particle.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, particle.r, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawStar(x, y, radius, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let index = 0; index < 10; index += 1) {
    const angle = -Math.PI / 2 + (index * Math.PI) / 5;
    const r = index % 2 ? radius * 0.46 : radius;
    const px = x + Math.cos(angle) * r;
    const py = y + Math.sin(angle) * r;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
}

function drawPlayerGuides(player) {
  ctx.save();
  ctx.fillStyle = player.color;
  [player.head, player.shoulder, player.arms.left.hand, player.arms.right.hand].forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.strokeStyle = player.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(player.shoulder.x, player.shoulder.y);
  ctx.lineTo(player.arms.right.hand.x, player.arms.right.hand.y);
  ctx.moveTo(player.shoulder.x, player.shoulder.y);
  ctx.lineTo(player.arms.left.hand.x, player.arms.left.hand.y);
  ctx.stroke();
  ctx.restore();
}

function drawDebug(poses) {
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  debugCtx.drawImage(video, 0, 0, debugCanvas.width, debugCanvas.height);
  poses.forEach((landmarks, poseIndex) => {
    const color = palette[poseIndex % palette.length];
    debugCtx.fillStyle = color;
    [
      landmarkNames.nose,
      landmarkNames.leftShoulder,
      landmarkNames.rightShoulder,
      landmarkNames.leftWrist,
      landmarkNames.rightWrist,
    ].forEach((index) => {
      const point = landmarks[index];
      if (!goodLandmark(point, 0.2)) return;
      debugCtx.beginPath();
      debugCtx.arc(point.x * debugCanvas.width, point.y * debugCanvas.height, 5, 0, Math.PI * 2);
      debugCtx.fill();
    });
  });
}

function roundRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function setStatus(text) {
  statusText.textContent = text;
}

debugToggle.addEventListener("click", () => {
  state.debug = !state.debug;
  debugToggle.setAttribute("aria-pressed", String(state.debug));
  cameraPanel.classList.toggle("is-visible", state.debug);
  if (!state.debug) {
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  }
});

startButton.addEventListener("click", startGame);
window.addEventListener("resize", resizeCanvas);
window.addEventListener("pointermove", (event) => {
  state.pointer.x = event.clientX;
  state.pointer.y = event.clientY;
  state.pointer.active = true;
});
window.addEventListener("pointerdown", (event) => {
  state.pointer.x = event.clientX;
  state.pointer.y = event.clientY;
  state.pointer.active = true;
  state.pointer.pressed = true;
});
window.addEventListener("pointerup", () => {
  state.pointer.pressed = false;
});

resizeCanvas();
seedScenery();
drawWorld(performance.now());
