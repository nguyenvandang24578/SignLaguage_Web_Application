import * as THREE      from 'three';
import { GLTFLoader }  from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── TỪ VỰNG ─────────────────────────────────────────────────────────────────
// videoFile : đặt trong  videos/
// glbFile   : đặt trong  glbs/
// Nếu chưa có file nào thì để null
const WORDS = [
  {
    label: "Ăn",
    topic: "Hành động",
    icon: "🍚",
    videoFile: "an.mp4",
    glbFile:   "an.glb",
    desc: "<strong>Mô tả:</strong> Đưa các ngón tay chụm lại gần miệng, lặp lại 2–3 lần."
  },
  {
    label: "Bàn chân",
    topic: "Cơ thể",
    icon: "🦶",
    videoFile: "ban_chan.mp4",
    glbFile:   "banchan.glb",
    desc: "<strong>Mô tả:</strong> Tay chỉ xuống bàn chân, vẽ đường bao quanh."
  },
  {
    label: "Bạn",
    topic: "Đại từ",
    icon: "🤝",
    videoFile: "ban.mp4",
    glbFile:   "ban.glb",
    desc: "<strong>Mô tả:</strong> Hai tay móc ngón vào nhau và lắc nhẹ."
  }
];
// ─────────────────────────────────────────────────────────────────────────────

const VIDEO_DIR = "./videos/";
const GLB_DIR   = "./glbs/";

// ── Build cards ───────────────────────────────────────────────────────────────
const grid = document.getElementById("wordGrid");
WORDS.forEach((w, i) => {
  const card = document.createElement("div");
  card.className = "word-card";
  card.innerHTML = `
    <span class="icon">${w.icon}</span>
    <div class="label">${w.label}</div>
    <div class="topic">${w.topic}</div>
    <div class="hint">▶ Xem ký hiệu</div>`;
  card.onclick = () => openModal(i);
  grid.appendChild(card);
});

// ── Modal elements ────────────────────────────────────────────────────────────
const overlay   = document.getElementById("overlay");
const video     = document.getElementById("modalVideo");
const noVideo   = document.getElementById("noVideo");
const noGlb     = document.getElementById("noGlb");
const btnPlay   = document.getElementById("btnPlay");
const btnLoop   = document.getElementById("btnLoop");

// ── Three.js state ────────────────────────────────────────────────────────────
let renderer, scene, camera, controls, mixer, clock, animFrameId;
let autoRotate = true;
let animPlaying = true;

function initThree() {
  const canvas = document.getElementById("threeCanvas");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  resizeRenderer();

  scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0e);

  camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
  camera.position.set(0.019, 0.573, 3.365);   // ← từ Blender convert


  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(2, 4, 3);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0x88ccff, 0.4);
  dir2.position.set(-2, 1, -2);
  scene.add(dir2);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate    = true;
  controls.autoRotateSpeed = 1.5;
  controls.minDistance   = 0.5;
  controls.maxDistance   = 8;

  clock = new THREE.Clock();
  renderLoop();
}

function resizeRenderer() {
  const canvas = document.getElementById("threeCanvas");
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  if (camera) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function renderLoop() {
  animFrameId = requestAnimationFrame(renderLoop);
  const delta = clock.getDelta();
  if (mixer && animPlaying) mixer.update(delta);
  controls.update();
  renderer.render(scene, camera);
}

function clearGlbScene() {
  if (!scene) return;
  scene.children
    .filter(c => c.userData.isSign)
    .forEach(c => { scene.remove(c); c.traverse(o => { if (o.isMesh) { o.geometry.dispose(); } }); });
  if (mixer) { mixer.stopAllAction(); mixer = null; }
}

function loadGlb(glbFile) {
  clearGlbScene();
  noGlb.classList.remove("show");

  const loader = new GLTFLoader();
  loader.load(
    GLB_DIR + glbFile,
    (gltf) => {
      const model = gltf.scene;
      model.userData.isSign = true;

      // KHÔNG auto-scale, giữ nguyên kích thước từ Blender
      scene.add(model);

      // Set camera đúng với thông số Blender (export_yup=True nên Y↔Z đã swap)
      // Blender: X=0.019, Y=-3.365, Z=0.573
      // Three.js (Y-up): X=0.019, Y=0.573, Z=3.365
      camera.position.set(0.019, 0.573, 3.365);
      controls.target.set(0, 0.5, 0);
      controls.update();

      // Play animation
      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model);
        const action = mixer.clipAction(gltf.animations[0]);
        action.setLoop(THREE.LoopRepeat, Infinity);
        action.play();
        animPlaying = true;
        document.getElementById("btnPlayAnim").textContent = "⏸ Animation: Bật";
        document.getElementById("btnPlayAnim").classList.add("on");
      }
    },
    undefined,
    () => {
      document.getElementById("noGlbPath").textContent = "glbs/" + glbFile;
      noGlb.classList.add("show");
    }
  );
}

// ── Tab switch ────────────────────────────────────────────────────────────────
let currentTab  = "video";
let currentWord = null;

window.switchTab = function(tab) {
  currentTab = tab;
  document.getElementById("tabVideo").classList.toggle("active", tab === "video");
  document.getElementById("tabGlb").classList.toggle("active",   tab === "glb");
  document.getElementById("videoPanel").style.display = tab === "video" ? "block" : "none";
  document.getElementById("glbPanel").style.display   = tab === "glb"   ? "block" : "none";
  document.getElementById("videoControls").style.display = tab === "video" ? "flex" : "none";
  document.getElementById("glbControls").style.display   = tab === "glb"   ? "flex" : "none";

  if (tab === "video") {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    video.play().catch(()=>{});
  } else {
    video.pause();
    if (!renderer) initThree(); else renderLoop();
    if (currentWord?.glbFile) loadGlb(currentWord.glbFile);
  }
};

// ── Open / Close modal ────────────────────────────────────────────────────────
function openModal(idx) {
  const w = currentWord = WORDS[idx];
  document.getElementById("modalTitle").textContent = w.label;
  document.getElementById("modalTopic").textContent = w.topic;
  document.getElementById("modalDesc").innerHTML    = w.desc;

  // Default tab = video
  switchTab("video");
  document.getElementById("tabVideo").classList.add("active");
  document.getElementById("tabGlb").classList.remove("active");

  // Load video
  noVideo.classList.remove("show");
  video.style.display = "block";
  video.src = VIDEO_DIR + w.videoFile;
  video.load();
  video.playbackRate = 1;
  video.loop = true;
  setActiveSpeed(1);

  video.onerror = () => {
    video.style.display = "none";
    document.getElementById("noVideoPath").textContent = "videos/" + w.videoFile;
    noVideo.classList.add("show");
  };

  overlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  overlay.classList.remove("active");
  document.body.style.overflow = "";
  video.pause(); video.src = "";
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  clearGlbScene();
  currentWord = null;
}

document.getElementById("closeBtn").onclick = closeModal;
overlay.addEventListener("click", e => { if (e.target === overlay) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

// ── Video controls ────────────────────────────────────────────────────────────
btnPlay.onclick = () => {
  if (video.paused) { video.play(); btnPlay.textContent = "⏸ Tạm dừng"; }
  else              { video.pause(); btnPlay.textContent = "▶ Phát"; }
};
video.addEventListener("play",  () => { btnPlay.textContent = "⏸ Tạm dừng"; });
video.addEventListener("pause", () => { btnPlay.textContent = "▶ Phát"; });

btnLoop.onclick = () => {
  video.loop = !video.loop;
  btnLoop.textContent = video.loop ? "🔁 Lặp: Bật" : "🔁 Lặp: Tắt";
  btnLoop.classList.toggle("on", video.loop);
};

function setActiveSpeed(s) {
  document.querySelectorAll("[data-speed]").forEach(b => {
    const on = parseFloat(b.dataset.speed) === s;
    b.classList.toggle("on", on);
  });
}
document.querySelectorAll("[data-speed]").forEach(b => {
  b.onclick = () => { video.playbackRate = parseFloat(b.dataset.speed); setActiveSpeed(parseFloat(b.dataset.speed)); };
});

// ── GLB controls ──────────────────────────────────────────────────────────────
document.getElementById("btnAutoRotate").onclick = () => {
  autoRotate = !autoRotate;
  if (controls) controls.autoRotate = autoRotate;
  const btn = document.getElementById("btnAutoRotate");
  btn.textContent = `↻ Auto-rotate: ${autoRotate ? "Bật" : "Tắt"}`;
  btn.classList.toggle("on", autoRotate);
};

document.getElementById("btnPlayAnim").onclick = () => {
  animPlaying = !animPlaying;
  const btn = document.getElementById("btnPlayAnim");
  btn.textContent = `⏸ Animation: ${animPlaying ? "Bật" : "Tắt"}`;
  btn.classList.toggle("on", animPlaying);
};

document.getElementById("btnResetCam").onclick = () => {
  if (!camera || !controls) return;
  camera.position.set(0.019, 0.573, 3.365);
  controls.target.set(0, 0.3, 0);
  controls.update();
};

window.addEventListener("resize", resizeRenderer);