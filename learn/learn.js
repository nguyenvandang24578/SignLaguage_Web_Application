/**
 * SignVN — learn.js
 * Vietnamese Sign Language Learning App
 */

import * as THREE        from 'three';
import { GLTFLoader }    from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadGRU, pushKeypointsAndPredict, resetGRUBuffer, resetGRUBufferAndSkip } from './models/gru_inference.js';
import {
  HandLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';
await loadGRU("./models/gru.onnx");

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════════════════════ */
const HF_BASE   = 'https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main';
const VIDEO_DIR = `${HF_BASE}/learn/videos/`;
const GLB_DIR   = `${HF_BASE}/learn/glbs/`;

/* ═══════════════════════════════════════════════════════════════════════════
   TOPIC DATA
   ═══════════════════════════════════════════════════════════════════════════ */
const TOPICS = [
  {
    id: 'bo-phan-co-the',
    label: 'Bộ phận cơ thể',
    icon: '🫀', iconBg: '#e3edf8', level: 'Cơ bản',
    desc: 'Học các ký hiệu chỉ các bộ phận trên cơ thể con người.',
    words: [
      { label: 'Bàn chân', videoFile: 'banchan.mp4',  glbFile: 'banchan.glb',  desc: '<strong>Mô tả:</strong> Tay chỉ xuống bàn chân, vẽ đường bao quanh.' },
      { label: 'Cổ',       videoFile: 'co.mp4',        glbFile: 'co.glb',       desc: '<strong>Mô tả:</strong> Tay đặt nhẹ lên cổ, vẽ vòng tròn.' },
      { label: 'Đầu',      videoFile: 'dau.mp4',       glbFile: 'dau.glb',      desc: '<strong>Mô tả:</strong> Đặt tay lên đỉnh đầu, vỗ nhẹ 2 lần.' },
      { label: 'Má',       videoFile: 'ma.mp4',        glbFile: 'ma.glb',       desc: '<strong>Mô tả:</strong> Chạm ngón trỏ vào má phải.' },
      { label: 'Mắt',      videoFile: 'mat.mp4',       glbFile: 'mat.glb',      desc: '<strong>Mô tả:</strong> Trỏ hai ngón vào mắt.' },
      { label: 'Miệng',    videoFile: 'mieng.mp4',     glbFile: 'mieng.glb',    desc: '<strong>Mô tả:</strong> Vẽ vòng tròn quanh miệng.' },
      { label: 'Mũi',      videoFile: 'mui.mp4',       glbFile: 'mui.glb',      desc: '<strong>Mô tả:</strong> Chạm ngón trỏ vào đầu mũi.' },
    ],
  },
  {
    id: 'hanh-dong',
    label: 'Hành động & Vận động',
    icon: '🏃', iconBg: '#fff0e3', level: 'Cơ bản',
    desc: 'Các ký hiệu về hoạt động thể chất và vận động hàng ngày.',
    words: [
      { label: 'Ăn',   videoFile: 'an.mp4',    glbFile: 'an.glb',    desc: '<strong>Mô tả:</strong> Đưa các ngón tay chụm lại gần miệng, lặp lại 2–3 lần.' },
      { label: 'Chạy', videoFile: 'chay.mp4',  glbFile: 'chay.glb',  desc: '<strong>Mô tả:</strong> Hai tay đánh nhanh qua lại như chạy.' },
      { label: 'Đi',   videoFile: 'di.mp4',    glbFile: 'di.glb',    desc: '<strong>Mô tả:</strong> Hai ngón trỏ và giữa mô phỏng chân bước đi.' },
      { label: 'Đứng', videoFile: 'dung.mp4',  glbFile: 'dung.glb',  desc: '<strong>Mô tả:</strong> Hai ngón đứng thẳng trên lòng bàn tay kia.' },
      { label: 'Nằm',  videoFile: 'nam.mp4',   glbFile: 'nam.glb',   desc: '<strong>Mô tả:</strong> Tay nằm ngang, ngón trỏ và giữa duỗi ra.' },
      { label: 'Ngồi', videoFile: 'ngoi.mp4',  glbFile: 'ngoi.glb',  desc: '<strong>Mô tả:</strong> Hai ngón gập xuống trên lòng bàn tay kia.' },
      { label: 'Ngủ',  videoFile: 'ngu.mp4',   glbFile: 'ngu.glb',   desc: '<strong>Mô tả:</strong> Nghiêng đầu, áp má vào lòng bàn tay.' },
      { label: 'Nhảy', videoFile: 'nhay.mp4',  glbFile: 'nhay.glb',  desc: '<strong>Mô tả:</strong> Hai ngón bật lên từ lòng bàn tay.' },
    ],
  },
  {
    id: 've-sinh',
    label: 'Vệ sinh & Sinh hoạt',
    icon: '🧴', iconBg: '#ede8f8', level: 'Cơ bản',
    desc: 'Ký hiệu về các hoạt động vệ sinh và sinh hoạt hàng ngày.',
    words: [
      { label: 'Chải đầu',    videoFile: 'chaidau.mp4',   glbFile: 'chaidau.glb',   desc: '<strong>Mô tả:</strong> Tay chải nhẹ qua tóc từ trên xuống.' },
      { label: 'Đánh răng',   videoFile: 'danhrang.mp4',  glbFile: 'danhrang.glb',  desc: '<strong>Mô tả:</strong> Ngón trỏ mô phỏng bàn chải đánh răng.' },
      { label: 'Đi vệ sinh',  videoFile: 'divesinh.mp4',  glbFile: 'divesinh.glb',  desc: '<strong>Mô tả:</strong> Chữ V di chuyển từ trên xuống dưới.' },
      { label: 'Gội đầu',     videoFile: 'goidau.mp4',    glbFile: 'goidau.glb',    desc: '<strong>Mô tả:</strong> Hai tay gội đầu từ trước ra sau.' },
      { label: 'Mặc quần áo', videoFile: 'macquanao.mp4', glbFile: 'macquanao.glb', desc: '<strong>Mô tả:</strong> Hai tay mô phỏng mặc quần áo vào người.' },
      { label: 'Rửa chân',    videoFile: 'ruachan.mp4',   glbFile: 'ruachan.glb',   desc: '<strong>Mô tả:</strong> Tay xoa vào chân như đang rửa.' },
      { label: 'Rửa mặt',     videoFile: 'ruamat.mp4',    glbFile: 'ruamat.glb',    desc: '<strong>Mô tả:</strong> Hai tay xoa lên mặt như rửa mặt.' },
      { label: 'Rửa tay',     videoFile: 'ruatay.mp4',    glbFile: 'ruatay.glb',    desc: '<strong>Mô tả:</strong> Hai tay xoa vào nhau.' },
    ],
  },
  {
    id: 'trang-phuc',
    label: 'Trang phục & Phụ kiện',
    icon: '👗', iconBg: '#fce8ed', level: 'Trung cấp',
    desc: 'Học ký hiệu về các loại quần áo và phụ kiện cá nhân.',
    words: [
      { label: 'Bít tất',      videoFile: 'bittat.mp4',     glbFile: 'bittat.glb',     desc: '<strong>Mô tả:</strong> Tay chỉ vào chân và vẽ vòng quanh cổ chân.' },
      { label: 'Cặp tóc',      videoFile: 'captoc.mp4',     glbFile: 'captoc.glb',     desc: '<strong>Mô tả:</strong> Hai ngón kẹp vào tóc.' },
      { label: 'Đôi dép',      videoFile: 'doidep.mp4',     glbFile: 'doidep.glb',     desc: '<strong>Mô tả:</strong> Tay vỗ nhẹ vào mu bàn chân.' },
      { label: 'Găng tay',     videoFile: 'gangtay.mp4',    glbFile: 'gangtay.glb',    desc: '<strong>Mô tả:</strong> Tay mô phỏng kéo găng tay lên.' },
      { label: 'Khăn mặt',     videoFile: 'khanmat.mp4',    glbFile: 'khanmat.glb',    desc: '<strong>Mô tả:</strong> Hai tay mô phỏng lau mặt bằng khăn.' },
      { label: 'Kính',         videoFile: 'kinh.mp4',       glbFile: 'kinh.glb',       desc: '<strong>Mô tả:</strong> Hai ngón trỏ và cái tạo vòng tròn quanh mắt.' },
      { label: 'Lược',         videoFile: 'luoc.mp4',       glbFile: 'luoc.glb',       desc: '<strong>Mô tả:</strong> Tay chải tóc bằng ngón tay như dùng lược.' },
      { label: 'Mũ lưỡi trai', videoFile: 'muluoitrai.mp4', glbFile: 'muluoitrai.glb', desc: '<strong>Mô tả:</strong> Tay mô phỏng đội mũ lưỡi trai lên đầu.' },
      { label: 'Nón',          videoFile: 'non.mp4',        glbFile: 'non.glb',        desc: '<strong>Mô tả:</strong> Tay tạo hình nón lá trên đầu.' },
    ],
  },
  {
    id: 'suc-khoe',
    label: 'Sức khỏe & Cảm xúc',
    icon: '💪', iconBg: '#e3f0f8', level: 'Trung cấp',
    desc: 'Ký hiệu về tình trạng sức khỏe và cảm xúc.',
    words: [
      { label: 'Béo',       videoFile: 'beo.mp4',      glbFile: 'beo.glb',      desc: '<strong>Mô tả:</strong> Hai tay phồng ra hai bên.' },
      { label: 'Cao',       videoFile: 'cao.mp4',      glbFile: 'cao.glb',      desc: '<strong>Mô tả:</strong> Tay giơ thẳng lên cao.' },
      { label: 'Cười',      videoFile: 'cuoi.mp4',     glbFile: 'cuoi.glb',     desc: '<strong>Mô tả:</strong> Hai ngón cái kéo góc miệng lên.' },
      { label: 'Gầy',       videoFile: 'gay.mp4',      glbFile: 'gay.glb',      desc: '<strong>Mô tả:</strong> Hai tay dẹp sát vào nhau.' },
      { label: 'Khóc',      videoFile: 'khoc.mp4',     glbFile: 'khoc.glb',     desc: '<strong>Mô tả:</strong> Ngón trỏ vẽ đường nước mắt xuống má.' },
      { label: 'Khỏe mạnh', videoFile: 'khoemanh.mp4', glbFile: 'khoemanh.glb', desc: '<strong>Mô tả:</strong> Hai tay nắm đấm chạm vào ngực.' },
      { label: 'Mệt mỏi',   videoFile: 'metmoi.mp4',   glbFile: 'metmoi.glb',   desc: '<strong>Mô tả:</strong> Vai xuôi, hai tay buông thõng.' },
      { label: 'Niềm vui',  videoFile: 'niemvui.mp4',  glbFile: 'niemvui.glb',  desc: '<strong>Mô tả:</strong> Hai tay di chuyển lên xuống trước ngực.' },
      { label: 'Sức khỏe',  videoFile: 'suckhoe.mp4',  glbFile: 'suckhoe.glb',  desc: '<strong>Mô tả:</strong> Tay nắm đấm gõ nhẹ lên ngực.' },
    ],
  },
  {
    id: 'gia-dinh',
    label: 'Gia đình & Xưng hô',
    icon: '👨‍👩‍👧', iconBg: '#fff0e3', level: 'Cơ bản',
    desc: 'Ký hiệu về các thành viên gia đình và cách xưng hô.',
    words: [
      { label: 'Bạn',     videoFile: 'ban.mp4',    glbFile: 'ban.glb',    desc: '<strong>Mô tả:</strong> Ngón trỏ chỉ về phía người đối diện.' },
      { label: 'Bé gái',  videoFile: 'begai.mp4',  glbFile: 'begai.glb',  desc: '<strong>Mô tả:</strong> Tay mô phỏng bé nhỏ + ký hiệu gái.' },
      { label: 'Bé trai', videoFile: 'betrai.mp4', glbFile: 'betrai.glb', desc: '<strong>Mô tả:</strong> Tay mô phỏng bé nhỏ + ký hiệu trai.' },
      { label: 'Bố',      videoFile: 'bo.mp4',     glbFile: 'bo.glb',     desc: '<strong>Mô tả:</strong> Đặt ngón cái lên trán.' },
      { label: 'Em trai', videoFile: 'emtrai.mp4', glbFile: 'emtrai.glb', desc: '<strong>Mô tả:</strong> Ngón trỏ chỉ về phía trước rồi hạ xuống.' },
    ],
  },
  {
    id: 'giao-tiep',
    label: 'Giao tiếp & Từ vựng',
    icon: '💬', iconBg: '#e8f4e3', level: 'Cơ bản',
    desc: 'Ký hiệu chào hỏi và từ vựng giao tiếp thông dụng.',
    words: [
      { label: 'Bao nhiêu', videoFile: 'baonhieu.mp4', glbFile: 'baonhieu.glb', desc: '<strong>Mô tả:</strong> Hai tay mở rộng, lòng bàn tay ngửa lên.' },
      { label: 'Chào',      videoFile: 'chao.mp4',     glbFile: 'chao.glb',     desc: '<strong>Mô tả:</strong> Giơ tay lên vẫy nhẹ sang ngang.' },
      { label: 'Sách',      videoFile: 'sach.mp4',     glbFile: 'sach.glb',     desc: '<strong>Mô tả:</strong> Hai tay mô phỏng mở sách ra.' },
    ],
  },
];

/* ═══════════════════════════════════════════════════════════════════════════
   SESSION PROGRESS
   ═══════════════════════════════════════════════════════════════════════════ */
const seenSet = new Set();

function markSeen(tIdx, wIdx) {
  seenSet.add(`${tIdx}-${wIdx}`);
  updateProgressBar(tIdx);
  renderLessonList(tIdx);
}

function isSeen(tIdx, wIdx) {
  return seenSet.has(`${tIdx}-${wIdx}`);
}

function getProgress(tIdx) {
  const total = TOPICS[tIdx].words.length;
  let done = 0;
  for (let i = 0; i < total; i++) if (isSeen(tIdx, i)) done++;
  return { done, total, pct: Math.round((done / total) * 100) };
}

function updateProgressBar(tIdx) {
  const bar   = document.getElementById('progressFill');
  const label = document.getElementById('progressPct');
  if (!bar || !label) return;
  const { pct } = getProgress(tIdx);
  bar.style.width   = `${pct}%`;
  label.textContent = `${pct}%`;
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP STATE
   ═══════════════════════════════════════════════════════════════════════════ */
let activeTopicIdx  = 0;
let activeLessonIdx = 0;
let currentViewTab  = 'video'; // 'video' | 'glb'

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR RENDERING
   ═══════════════════════════════════════════════════════════════════════════ */
function renderSidebar() {
  const topicNav = document.getElementById('topicNav');
  if (!topicNav) return;

  /* ── Search bar ── */
  const sidebar = document.querySelector('.sidebar');
  if (sidebar && !document.getElementById('sidebarSearch')) {
    const searchWrap = document.createElement('div');
    searchWrap.className = 'sidebar-search-wrap';
    searchWrap.innerHTML = `
      <div class="sidebar-search-box">
        <svg class="sidebar-search-icon" width="15" height="15" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <input id="sidebarSearch" type="text" placeholder="Tìm từ vựng…"
               autocomplete="off" spellcheck="false" />
        <button id="sidebarSearchClear" class="sidebar-search-clear" style="display:none">✕</button>
      </div>
      <div id="searchResults" class="search-results" style="display:none"></div>`;
    sidebar.insertBefore(searchWrap, sidebar.querySelector('.section-label'));

    const input  = document.getElementById('sidebarSearch');
    const clear  = document.getElementById('sidebarSearchClear');
    const results= document.getElementById('searchResults');

    input.addEventListener('input', () => {
      const q = input.value.trim().toLowerCase();
      clear.style.display = q ? '' : 'none';
      if (!q) {
        results.style.display = 'none';
        topicNav.style.display = '';
        document.querySelector('.sidebar .section-label').style.display = '';
        return;
      }
      topicNav.style.display = 'none';
      document.querySelector('.sidebar .section-label').style.display = 'none';
      renderSearchResults(q, results);
      results.style.display = '';
    });

    clear.addEventListener('click', () => {
      input.value = '';
      clear.style.display = 'none';
      results.style.display = 'none';
      topicNav.style.display = '';
      document.querySelector('.sidebar .section-label').style.display = '';
      input.focus();
    });
  }

  /* ── Topic list ── */
  topicNav.innerHTML = '';
  TOPICS.forEach((t, i) => {
    const item = document.createElement('div');
    item.className   = `topic-nav-item${i === 0 ? ' active' : ''}`;
    item.role        = 'listitem';
    item.tabIndex    = 0;
    item.innerHTML   = `
      <div class="topic-nav-icon" style="background:${t.iconBg}">${t.icon}</div>
      <div class="topic-nav-info">
        <div class="topic-nav-name">${t.label}</div>
        <div class="topic-nav-count">${t.words.length} bài học</div>
      </div>`;
    item.onclick   = () => selectTopic(i);
    item.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') selectTopic(i); };
    topicNav.appendChild(item);
  });
}

/* ── Search across all topics ─────────────────────────────────────────────── */
function renderSearchResults(q, container) {
  const hits = [];
  TOPICS.forEach((t, tIdx) => {
    t.words.forEach((w, wIdx) => {
      if (w.label.toLowerCase().includes(q)) hits.push({ t, tIdx, w, wIdx });
    });
  });

  if (!hits.length) {
    container.innerHTML = `<div class="search-empty">Không tìm thấy từ "<strong>${q}</strong>"</div>`;
    return;
  }

  container.innerHTML = hits.map(({ t, tIdx, w, wIdx }) => `
    <div class="search-result-item" data-tidx="${tIdx}" data-widx="${wIdx}">
      <div class="search-result-icon" style="background:${t.iconBg}">${t.icon}</div>
      <div class="search-result-body">
        <div class="search-result-word">${w.label}</div>
        <div class="search-result-topic">${t.label}</div>
      </div>
    </div>`).join('');

  container.querySelectorAll('.search-result-item').forEach(el => {
    el.addEventListener('click', () => {
      const tIdx = +el.dataset.tidx;
      const wIdx = +el.dataset.widx;
      /* clear search */
      const input = document.getElementById('sidebarSearch');
      const clear = document.getElementById('sidebarSearchClear');
      input.value = '';
      clear.style.display = 'none';
      container.style.display = 'none';
      document.getElementById('topicNav').style.display = '';
      document.querySelector('.sidebar .section-label').style.display = '';
      /* navigate */
      selectTopic(tIdx);
      selectLesson(tIdx, wIdx);
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   SELECT TOPIC
   ═══════════════════════════════════════════════════════════════════════════ */
function selectTopic(idx) {
  activeTopicIdx  = idx;
  activeLessonIdx = 0;

  document.querySelectorAll('.topic-nav-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  renderFeaturedCard();
  selectLesson(idx, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
   FEATURED CARD
   ═══════════════════════════════════════════════════════════════════════════ */
function renderFeaturedCard() {
  const t   = TOPICS[activeTopicIdx];
  const { pct } = getProgress(activeTopicIdx);
  const card = document.getElementById('featuredCard');
  if (!card) return;

  card.innerHTML = `
    <!-- INFO BAR -->
    <div class="feat-info-bar">
      <div>
        <div class="feat-badge">${t.level}</div>
        <div class="feat-title">${t.label}</div>
        <div class="feat-desc">${t.desc}</div>
      </div>
      <div class="feat-info-right">
        <div class="feat-progress-label">
          <span>Tiến độ của bạn</span>
          <span id="progressPct">${pct}%</span>
        </div>
        <div class="feat-progress-bar">
          <div class="feat-progress-fill" id="progressFill" style="width:${pct}%"></div>
        </div>
        <div class="feat-btn-row">
          <button class="feat-btn feat-btn-primary" id="featContinueBtn">Tiếp tục học ▶</button>
          <button class="feat-btn feat-btn-secondary" id="featPracticeBtn">🔤 Luyện ghép chữ cái</button>
        </div>
      </div>
    </div>

    <!-- MAIN: viewer + lesson list -->
    <div class="feat-main">

      <!-- INLINE VIEWER -->
      <div class="inline-viewer">

        <!-- Screen -->
        <div class="viewer-screen" id="viewerScreen">
          <video id="inlineVideo" class="inline-video" muted playsinline></video>

          <!-- No-video placeholder -->
          <div class="no-media" id="noVideo">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.3">
              <rect x="2" y="6" width="20" height="12" rx="2"/>
              <path d="M10 10l4 2-4 2v-4z"/>
              <line x1="3" y1="3" x2="21" y2="21"/>
            </svg>
            <div class="no-media-title">Chưa có video</div>
            <span>Đặt file vào: <code id="noVideoPath"></code></span>
          </div>

          <!-- GLB canvas -->
          <canvas id="threeCanvas" style="display:none; position:absolute; inset:0;
            width:100%; height:100%; touch-action:none;"></canvas>
          <div class="glb-hint" id="glbHint" style="display:none">
            🖱 Kéo để xoay · Scroll để zoom
          </div>

          <!-- No-glb placeholder -->
          <div class="no-media" id="noGlb">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.4">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              <line x1="3" y1="3" x2="21" y2="21"/>
            </svg>
            <span>Chưa có file GLB<br><code id="noGlbPath"></code></span>
          </div>
        </div>

        <!-- Tab bar -->
        <div class="viewer-tabs">
          <button class="vtab-btn active" id="vtabVideo"
                  onclick="window.switchViewTab('video')">▶ Video</button>
          <button class="vtab-btn"        id="vtabGlb"
                  onclick="window.switchViewTab('glb')">⬡ 3D Tương tác</button>
        </div>

        <!-- Controls -->
        <div class="viewer-controls">
          <div id="videoCtrl">
            <button class="ctrl-btn" id="btnPlay">▶ Phát</button>
            <button class="ctrl-btn on" id="btnLoop">🔁 Lặp: Bật</button>
            <div class="sep"></div>
            <span class="speed-label">Tốc độ:</span>
            <button class="ctrl-btn" data-speed="0.5">0.5×</button>
            <button class="ctrl-btn on" data-speed="1">1×</button>
            <button class="ctrl-btn" data-speed="1.5">1.5×</button>
          </div>
          <div id="glbCtrl">
            <button class="ctrl-btn on" id="btnAutoRotate">↻ Auto-rotate: Bật</button>
            <button class="ctrl-btn on" id="btnPlayAnim">⏸ Animation: Bật</button>
            <button class="ctrl-btn"   id="btnResetCam">⌖ Reset góc nhìn</button>
          </div>
        </div>

        <!-- Description -->
        <div class="viewer-desc" id="viewerDesc">Chọn một bài học để bắt đầu.</div>
      </div>

      <!-- LESSON LIST -->
      <div class="feat-lessons">
        <div class="feat-lessons-header">
          <span>Bài học trong chủ đề</span>
          <button class="feat-see-all">Xem tất cả</button>
        </div>
        <div class="lesson-list" id="lessonList"></div>
      </div>

    </div>`;

  /* Wire controls */
  wireVideoControls();
  wireGlbControls();

  document.getElementById('featContinueBtn').onclick = () => {
    const idx = TOPICS[activeTopicIdx].words.findIndex((_, i) => !isSeen(activeTopicIdx, i));
    selectLesson(activeTopicIdx, idx >= 0 ? idx : 0);
  };

  document.getElementById('featPracticeBtn').onclick = () => {
    openPracticeWordModal(activeTopicIdx);
  };

  renderLessonList(activeTopicIdx);
}

/* ═══════════════════════════════════════════════════════════════════════════
   LESSON LIST
   ═══════════════════════════════════════════════════════════════════════════ */
function renderLessonList(tIdx) {
  const list = document.getElementById('lessonList');
  if (!list) return;
  list.innerHTML = '';

  TOPICS[tIdx].words.forEach((w, i) => {
    const seen   = isSeen(tIdx, i);
    const active = i === activeLessonIdx;
    const item   = document.createElement('div');
    item.className = `lesson-item${seen ? ' done' : ''}${active ? ' active' : ''}`;
    item.innerHTML = `
      <div class="lesson-num">${seen ? '✓' : i + 1}</div>
      <div class="lesson-name">${w.label}</div>
      <span class="lesson-icon">${seen ? '✓' : active ? '▶' : '›'}</span>`;
    item.onclick = () => selectLesson(tIdx, i);
    list.appendChild(item);
  });

  /* Complete banner */
  document.querySelector('.complete-banner')?.remove();
  const { done, total } = getProgress(tIdx);
  if (done === total && total > 0) {
    const banner = document.createElement('div');
    banner.className = 'complete-banner';
    banner.innerHTML = `🎉 <span>Hoàn thành tất cả ${total} bài học!</span>`;
    list.after(banner);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SELECT LESSON
   ═══════════════════════════════════════════════════════════════════════════ */
function selectLesson(tIdx, wIdx) {
  activeLessonIdx = wIdx;
  const w = TOPICS[tIdx].words[wIdx];

  markSeen(tIdx, wIdx);
  renderLessonList(tIdx);

  const desc = document.getElementById('viewerDesc');
  if (desc) desc.innerHTML = w.desc;

  if (currentViewTab === 'video') {
    loadInlineVideo(w);
  } else {
    loadGlb(w.glbFile);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   INLINE VIDEO
   ═══════════════════════════════════════════════════════════════════════════ */
function loadInlineVideo(w) {
  const vid   = document.getElementById('inlineVideo');
  const noVid = document.getElementById('noVideo');
  if (!vid) return;

  noVid.classList.remove('show');
  vid.style.display = 'block';
  vid.src  = VIDEO_DIR + w.videoFile;
  vid.loop = document.getElementById('btnLoop')?.classList.contains('on') ?? true;
  vid.load();
  vid.pause();

  showVideoPlayOverlay(vid);

  vid.onerror = () => {
    vid.style.display = 'none';
    const p = document.getElementById('noVideoPath');
    if (p) p.textContent = `videos/${w.videoFile}`;
    noVid.classList.add('show');
  };
}

/* ── Video Play Overlay ───────────────────────────────────────────────────── */
function showVideoPlayOverlay(videoEl) {
  const screen = document.getElementById('viewerScreen');
  if (!screen) return;
  screen.querySelector('.vid-play-overlay')?.remove();

  const overlay = document.createElement('div');
  overlay.className = 'vid-play-overlay';
  overlay.innerHTML = `
    <button class="vid-play-btn" aria-label="Phát video">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 5v14l11-7z"/>
      </svg>
    </button>`;

  overlay.querySelector('.vid-play-btn').onclick = (e) => {
    e.stopPropagation();
    overlay.remove();
    videoEl.play();
  };

  /* When video ends → show overlay again (unless looping) */
  videoEl.onended = () => {
    const loopBtn = document.getElementById('btnLoop');
    if (loopBtn?.classList.contains('on')) return;
    showVideoPlayOverlay(videoEl);
  };

  screen.appendChild(overlay);
}

/* ── Update play/pause button label ──────────────────────────────────────── */
function syncPlayBtn(paused) {
  const btn = document.getElementById('btnPlay');
  if (!btn) return;
  btn.textContent = paused ? '▶ Phát' : '⏸ Tạm dừng';
}

/* ═══════════════════════════════════════════════════════════════════════════
   VIEW TAB SWITCH  (exposed to window for inline onclick)
   ═══════════════════════════════════════════════════════════════════════════ */
window.switchViewTab = function (tab) {
  currentViewTab = tab;

  const vid      = document.getElementById('inlineVideo');
  const canvas   = document.getElementById('threeCanvas');
  const glbHint  = document.getElementById('glbHint');
  const videoCtrl = document.getElementById('videoCtrl');
  const glbCtrl  = document.getElementById('glbCtrl');
  const noGlb    = document.getElementById('noGlb');
  const noVid    = document.getElementById('noVideo');

  document.getElementById('vtabVideo')?.classList.toggle('active', tab === 'video');
  document.getElementById('vtabGlb')?.classList.toggle('active',   tab === 'glb');
  if (videoCtrl) videoCtrl.style.display = tab === 'video' ? 'flex' : 'none';
  if (glbCtrl)   glbCtrl.style.display   = tab === 'glb'   ? 'flex' : 'none';

  if (tab === 'video') {
    if (canvas)  canvas.style.display  = 'none';
    if (glbHint) glbHint.style.display = 'none';
    noGlb?.classList.remove('show');
    if (vid) { vid.style.display = 'block'; }
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  } else {
    if (vid) { vid.pause(); vid.style.display = 'none'; }
    noVid?.classList.remove('show');
    if (canvas)  canvas.style.display  = 'block';
    if (glbHint) glbHint.style.display = 'block';
    if (!renderer) initThree(); else startRenderLoop();
    const w = TOPICS[activeTopicIdx].words[activeLessonIdx];
    if (w?.glbFile) loadGlb(w.glbFile);
  }
};

/* ═══════════════════════════════════════════════════════════════════════════
   VIDEO CONTROLS
   ═══════════════════════════════════════════════════════════════════════════ */
function wireVideoControls() {
  const getVid  = () => document.getElementById('inlineVideo');
  const btnPlay = document.getElementById('btnPlay');
  const btnLoop = document.getElementById('btnLoop');

  if (btnPlay) {
    btnPlay.onclick = () => {
      const v = getVid(); if (!v) return;
      /* Remove overlay if present */
      document.querySelector('.vid-play-overlay')?.remove();
      if (v.paused) {
        v.play();
        syncPlayBtn(false);
      } else {
        v.pause();
        syncPlayBtn(true);
      }
    };
  }

  if (btnLoop) {
    /* Initialise: loop ON by default */
    const v = getVid();
    if (v) v.loop = true;

    btnLoop.onclick = () => {
      const v = getVid(); if (!v) return;
      v.loop = !v.loop;
      btnLoop.textContent = `🔁 Lặp: ${v.loop ? 'Bật' : 'Tắt'}`;
      btnLoop.classList.toggle('on', v.loop);
    };
  }

  document.querySelectorAll('[data-speed]').forEach(b => {
    b.onclick = () => {
      const v = getVid(); if (!v) return;
      const s = parseFloat(b.dataset.speed);
      v.playbackRate = s;
      document.querySelectorAll('[data-speed]').forEach(x => {
        x.classList.toggle('on', parseFloat(x.dataset.speed) === s);
      });
    };
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   THREE.JS
   ═══════════════════════════════════════════════════════════════════════════ */
let renderer, scene, camera, orbitCtrl, mixer, clock, animFrameId;
let autoRotate  = true;
let animPlaying = true;
let glbLoadId   = 0; // incremented each load; stale callbacks check against this

function initThree() {
  const canvas = document.getElementById('threeCanvas');
  if (!canvas) return;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  resizeRenderer();

  scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e1a);

  camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
  camera.position.set(0.019, 0.573, 3.365);

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  const dir1    = new THREE.DirectionalLight(0xffffff, 1.2);
  const dir2    = new THREE.DirectionalLight(0x88ccff, 0.4);
  dir1.position.set(2, 4, 3);
  dir2.position.set(-2, 1, -2);
  scene.add(ambient, dir1, dir2);

  orbitCtrl = new OrbitControls(camera, canvas);
  orbitCtrl.enableDamping   = true;
  orbitCtrl.dampingFactor   = 0.08;
  orbitCtrl.autoRotate      = true;
  orbitCtrl.autoRotateSpeed = 1.5;
  orbitCtrl.minDistance     = 0.5;
  orbitCtrl.maxDistance     = 8;

  clock = new THREE.Clock();
  startRenderLoop();
  window.addEventListener('resize', resizeRenderer);
}

function resizeRenderer() {
  const canvas = document.getElementById('threeCanvas');
  if (!canvas || !renderer) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  if (camera) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
}

function startRenderLoop() {
  if (animFrameId) return;
  function loop() {
    animFrameId = requestAnimationFrame(loop);
    if (mixer && animPlaying) mixer.update(clock.getDelta());
    orbitCtrl.update();
    renderer.render(scene, camera);
  }
  loop();
}

function clearGlbScene() {
  if (!scene) return;
  const toRemove = scene.children.filter(c => c.userData.isSign);
  toRemove.forEach(c => {
    scene.remove(c);
    c.traverse(o => {
      if (o.isMesh) {
        o.geometry.dispose();
        if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
        else if (o.material) o.material.dispose();
      }
    });
  });
  if (mixer) { mixer.stopAllAction(); mixer = null; }
}

function loadGlb(glbFile) {
  clearGlbScene();
  const noGlb = document.getElementById('noGlb');
  noGlb?.classList.remove('show');

  const myLoadId = ++glbLoadId; // capture current ID; stale callbacks will bail

  new GLTFLoader().load(
    GLB_DIR + glbFile,
    (gltf) => {
      if (myLoadId !== glbLoadId) return; // stale — a newer load already started

      const model = gltf.scene;
      model.userData.isSign = true;
      scene.add(model);

      camera.position.set(0.019, 0.573, 3.365);
      orbitCtrl.target.set(0, 0.5, 0);
      orbitCtrl.update();

      if (gltf.animations?.length) {
        mixer = new THREE.AnimationMixer(model);
        mixer.clipAction(gltf.animations[0]).setLoop(THREE.LoopRepeat, Infinity).play();
        animPlaying = true;
        const btn = document.getElementById('btnPlayAnim');
        if (btn) { btn.textContent = '⏸ Animation: Bật'; btn.classList.add('on'); }
      }
    },
    undefined,
    () => {
      if (myLoadId !== glbLoadId) return;
      const p = document.getElementById('noGlbPath');
      if (p) p.textContent = `glbs/${glbFile}`;
      noGlb?.classList.add('show');
    },
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   GLB CONTROLS
   ═══════════════════════════════════════════════════════════════════════════ */
function wireGlbControls() {
  const btnAR = document.getElementById('btnAutoRotate');
  const btnPA = document.getElementById('btnPlayAnim');
  const btnRC = document.getElementById('btnResetCam');

  if (btnAR) {
    btnAR.onclick = () => {
      autoRotate = !autoRotate;
      if (orbitCtrl) orbitCtrl.autoRotate = autoRotate;
      btnAR.textContent = `↻ Auto-rotate: ${autoRotate ? 'Bật' : 'Tắt'}`;
      btnAR.classList.toggle('on', autoRotate);
    };
  }
  if (btnPA) {
    btnPA.onclick = () => {
      animPlaying = !animPlaying;
      btnPA.textContent = `⏸ Animation: ${animPlaying ? 'Bật' : 'Tắt'}`;
      btnPA.classList.toggle('on', animPlaying);
    };
  }
  if (btnRC) {
    btnRC.onclick = () => {
      if (!camera || !orbitCtrl) return;
      camera.position.set(0.019, 0.573, 3.365);
      orbitCtrl.target.set(0, 0.3, 0);
      orbitCtrl.update();
    };
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   MEDIAPIPE HAND LANDMARKER
   ═══════════════════════════════════════════════════════════════════════════ */
let handLandmarker    = null;
let handDetectRunning = false;

async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  console.log('[MediaPipe] HandLandmarker ready');
}

/* ═══════════════════════════════════════════════════════════════════════════
   VIETNAMESE WORD SPLITTER
   ═══════════════════════════════════════════════════════════════════════════ */
const VIET_TONE_MAP = {
  'á':['a','sắc'],   'à':['a','huyền'], 'ả':['a','hỏi'], 'ã':['a','ngã'],  'ạ':['a','nặng'],
  'ắ':['ă','sắc'],   'ằ':['ă','huyền'], 'ẳ':['ă','hỏi'], 'ẵ':['ă','ngã'],  'ặ':['ă','nặng'],
  'ấ':['â','sắc'],   'ầ':['â','huyền'], 'ẩ':['â','hỏi'], 'ẫ':['â','ngã'],  'ậ':['â','nặng'],
  'é':['e','sắc'],   'è':['e','huyền'], 'ẻ':['e','hỏi'], 'ẽ':['e','ngã'],  'ẹ':['e','nặng'],
  'ế':['ê','sắc'],   'ề':['ê','huyền'], 'ể':['ê','hỏi'], 'ễ':['ê','ngã'],  'ệ':['ê','nặng'],
  'í':['i','sắc'],   'ì':['i','huyền'], 'ỉ':['i','hỏi'], 'ĩ':['i','ngã'],  'ị':['i','nặng'],
  'ó':['o','sắc'],   'ò':['o','huyền'], 'ỏ':['o','hỏi'], 'õ':['o','ngã'],  'ọ':['o','nặng'],
  'ố':['ô','sắc'],   'ồ':['ô','huyền'], 'ổ':['ô','hỏi'], 'ỗ':['ô','ngã'],  'ộ':['ô','nặng'],
  'ớ':['ơ','sắc'],   'ờ':['ơ','huyền'], 'ở':['ơ','hỏi'], 'ỡ':['ơ','ngã'],  'ợ':['ơ','nặng'],
  'ú':['u','sắc'],   'ù':['u','huyền'], 'ủ':['u','hỏi'], 'ũ':['u','ngã'],  'ụ':['u','nặng'],
  'ứ':['ư','sắc'],   'ừ':['ư','huyền'], 'ử':['ư','hỏi'], 'ữ':['ư','ngã'],  'ự':['ư','nặng'],
  'ý':['y','sắc'],   'ỳ':['y','huyền'], 'ỷ':['y','hỏi'], 'ỹ':['y','ngã'],  'ỵ':['y','nặng'],
};

const VIET_DIGRAPHS = ['ngh', 'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr'];

function splitWord(word) {
  const s = word.toLowerCase();
  const result = [];
  let i = 0;

  while (i < s.length) {
    const ch = s[i];
    if (ch === ' ') { i++; continue; }
    if (i + 2 < s.length && s.slice(i, i + 3) === 'ngh') {
      result.push('ngh'); i += 3; continue;
    }
    if (i + 1 < s.length) {
      const di = s.slice(i, i + 2);
      if (VIET_DIGRAPHS.includes(di)) {
        result.push(di); i += 2; continue;
      }
    }
    if (VIET_TONE_MAP[ch]) {
      result.push(...VIET_TONE_MAP[ch]);
      i++; continue;
    }
    result.push(ch);
    i++;
  }

  return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
   GRU LABEL MAPPING
   ═══════════════════════════════════════════════════════════════════════════ */
const SUPPORTED_GRU_LABELS = new Set([
  "a", "aw", "aa",
  "b", "c", "d", "dd",
  "e", "ee",
  "g", "h", "i", "k", "l", "m", "n", "o",
  "ow", "oo",
  "p", "q", "r", "s", "t",
  "u", "uw", "v", "x", "y",
  "sac", "hoi", "huyen", "nga", "nang",
]);

const CHAR_TO_GRU = {
  "ă": "aw",
  "â": "aa",
  "ê": "ee",
  "ô": "ow",
  "ơ": "oo",
  "ư": "uw",
  "đ": "dd",
  "sắc":   "sac",
  "hỏi":   "hoi",
  "nặng":  "nang",
  "ngã":   "nga",
  "huyền": "huyen",
  "a":"a", "b":"b", "c":"c", "d":"d",
  "e":"e", "g":"g", "h":"h", "i":"i", "k":"k",
  "l":"l", "m":"m", "n":"n", "o":"o", "p":"p",
  "q":"q", "r":"r", "s":"s", "t":"t", "u":"u",
  "v":"v", "x":"x", "y":"y",
};

function getCurrentModelType() {
  return "gru";
}

function getCurrentGRULabel() {
  const char = practiceState.chars[practiceState.currentCharIdx]?.toLowerCase();
  return CHAR_TO_GRU[char] ?? char ?? null;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PRACTICE MODE
   ═══════════════════════════════════════════════════════════════════════════ */
let practiceState = {
  active:         false,
  word:           '',
  chars:          [],
  currentCharIdx: 0,
  cameraStream:   null,
  topicIdx:       0,
};

/* ── Word selection modal ─────────────────────────────────────────────────── */
function isWordSupported(word) {
  return splitWord(word).every(ch => {
    const label = CHAR_TO_GRU[ch.toLowerCase()] ?? ch.toLowerCase();
    return SUPPORTED_GRU_LABELS.has(label);
  });
}

function openPracticeWordModal(tIdx) {
  const allWords   = TOPICS[tIdx].words.map(w => w.label);
  const words      = allWords.filter(isWordSupported);
  const topicLabel = TOPICS[tIdx].label;

  const overlay = document.createElement('div');
  overlay.className = 'practice-modal-overlay';
  overlay.id        = 'practiceModalOverlay';

  const wordGridHtml = words.length
    ? words.map(w => `<button class="practice-word-btn" data-word="${w}">${w}</button>`).join('')
    : `<p style="color:var(--text-muted);font-size:0.9rem;">Chủ đề này chưa có từ nào được hỗ trợ.</p>`;

  overlay.innerHTML = `
    <div class="practice-modal">
      <h2>🎯 Chọn từ để luyện tập</h2>
      <p>Chủ đề: <strong>${topicLabel}</strong> — Chọn 1 từ, hệ thống sẽ bật camera để bạn thực hành.</p>
      <div class="practice-word-grid">${wordGridHtml}</div>
      <button class="practice-modal-close">✕ Đóng</button>
    </div>`;

  document.body.appendChild(overlay);

  overlay.querySelectorAll('.practice-word-btn').forEach(btn => {
    btn.onclick = () => {
      closePracticeModal();
      startPracticeSession(tIdx, btn.dataset.word);
    };
  });
  overlay.querySelector('.practice-modal-close').onclick = closePracticeModal;
  overlay.addEventListener('click', e => { if (e.target === overlay) closePracticeModal(); });
}

function closePracticeModal() {
  document.getElementById('practiceModalOverlay')?.remove();
}

/* ── Start practice session ───────────────────────────────────────────────── */
function startPracticeSession(tIdx, word) {
  gruLoopGeneration++;

  if (practiceState.active) {
    gruDisplaying = false;
    stopCamera();
  }

  cachedCanvasW = 0;
  cachedCanvasH = 0;

  practiceState = {
    active:         true,
    word,
    chars:          splitWord(word),
    currentCharIdx: 0,
    cameraStream:   null,
    topicIdx:       tIdx,
  };

  const featCard = document.querySelector('.featured-card');
  if (featCard) featCard.style.display = 'none';

  renderPracticeView();
  startCamera();
  loadCharVideo(0);
}

/* ── Render practice view ─────────────────────────────────────────────────── */
function renderPracticeView() {
  document.getElementById('practiceView')?.remove();

  const { word, chars, currentCharIdx } = practiceState;
  const charChipsHtml = buildCharChips(chars, currentCharIdx);

  const view = document.createElement('div');
  view.className = 'practice-view';
  view.id        = 'practiceView';
  view.innerHTML = `
    <div class="practice-header">
      <div class="practice-header-left">
        <h2>Luyện tập: "${word}"</h2>
        <p>Thực hiện từng chữ cái theo thứ tự</p>
        <p style="font-size:.7rem;color:rgba(255,255,255,.5);margin-top:2px;">
          ⚠️ Luyện tập ghép chữ cái — khác với video từ hoàn chỉnh trong phần học
        </p>
        <div class="char-progress" id="charProgress">${charChipsHtml}</div>
      </div>
      <button class="practice-exit-btn" id="practiceExitBtn">✕ Thoát</button>
    </div>

    <div class="practice-body">
      <!-- Camera panel -->
      <div class="practice-panel">
        <div class="practice-panel-header">
          <span class="panel-dot red"></span> Camera của bạn
        </div>
        <div class="practice-cam-wrap">
          <video id="practiceCamera" autoplay playsinline muted></video>
          <canvas id="handCanvas"></canvas>

          <!-- Model inference counter -->
          <div class="gru-counter" id="gruCounter">
            <span class="gru-counter-bar" id="gruCounterBar"></span>
            <span class="gru-counter-label">
              <span id="gruFrameCount">0</span><span id="gruFrameMax">/30</span>
              <span id="gruModelBadge" class="gru-badge">GRU</span>
            </span>
          </div>

          <div class="cam-placeholder" id="camPlaceholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.5">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1
                       2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <span>Đang mở camera…</span>
          </div>
        </div>
      </div>

      <!-- Reference video panel -->
      <div class="practice-panel">
        <div class="practice-panel-header">
          <span class="panel-dot blue"></span>
          Ký hiệu mẫu — chữ "<span id="currentCharLabel">${chars[0]}</span>"
        </div>
        <div class="practice-vid-wrap">
          <video id="practiceRefVideo" autoplay loop muted playsinline></video>
        </div>
      </div>
    </div>

    <div class="practice-footer">
      <div class="practice-status">
        <span class="status-indicator waiting" id="statusDot"></span>
        <span id="statusText">Đang chờ bạn thực hiện chữ "<strong>${chars[0]}</strong>"…</span>
      </div>
      <button class="practice-manual-btn" id="practiceNextBtn">✅ Đúng rồi → Tiếp theo</button>
    </div>`;

  document.querySelector('.content-area').appendChild(view);

  document.getElementById('practiceExitBtn').onclick = exitPracticeSession;
  document.getElementById('practiceNextBtn').onclick = advanceChar;
}

/* ── Build char chip HTML ─────────────────────────────────────────────────── */
function buildCharChips(chars, currentIdx) {
  return chars.map((c, i) => {
    const cls = i < currentIdx ? 'done' : i === currentIdx ? 'active' : '';
    return `<div class="char-chip ${cls}">${c}</div>`;
  }).join('<div class="char-chip separator">·</div>');
}

/* ── Refresh char progress display ───────────────────────────────────────── */
function refreshCharProgress() {
  const container = document.getElementById('charProgress');
  if (!container) return;
  container.innerHTML = buildCharChips(practiceState.chars, practiceState.currentCharIdx);
}

/* ── Load reference video for current char ───────────────────────────────── */
function loadCharVideo(charIdx) {
  // Reset buffer VÀ skip 15 frame (~500ms) để counter thực sự hiển thị 0
  // trước khi bắt đầu thu thập lại — dù tay có còn trong frame hay không.
  // Điều này áp dụng cho cả trường hợp đúng (advance) lẫn bắt đầu session mới.
  resetGRUBufferAndSkip(15);
  setGruCounter(0, false, 30, 'GRU');
  gruDisplaying = false;

  const { chars } = practiceState;
  const char   = chars[charIdx];
  const refVid = document.getElementById('practiceRefVideo');
  if (!refVid) return;

  const gruLabel = CHAR_TO_GRU[char?.toLowerCase()] ?? char?.toLowerCase();
  refVid.src = VIDEO_DIR + gruLabel + '.mp4';
  refVid.load();
  refVid.play().catch(() => {});

  const label = document.getElementById('currentCharLabel');
  if (label) label.textContent = char;

  const statusText = document.getElementById('statusText');
  if (statusText) statusText.innerHTML = `Đang chờ bạn thực hiện chữ "<strong>${char}</strong>"…`;

  const statusDot = document.getElementById('statusDot');
  if (statusDot) statusDot.className = 'status-indicator waiting';
}

/* ── Advance to next character ────────────────────────────────────────────── */
function advanceChar() {
  const statusDot = document.getElementById('statusDot');
  if (statusDot) statusDot.className = 'status-indicator correct';

  const next = practiceState.currentCharIdx + 1;
  practiceState.currentCharIdx = next;
  refreshCharProgress();

  if (next >= practiceState.chars.length) {
    showWordSuccess();
    return;
  }
  loadCharVideo(next);
}

/* ── Get next supported word in topic ────────────────────────────────────── */
function getNextSupportedWord(tIdx, currentWord) {
  const words = TOPICS[tIdx].words;
  const currentIdx = words.findIndex(w => w.label === currentWord);
  for (let i = currentIdx + 1; i < words.length; i++) {
    if (isWordSupported(words[i].label)) return words[i].label;
  }
  // Wrap around from beginning
  for (let i = 0; i < currentIdx; i++) {
    if (isWordSupported(words[i].label)) return words[i].label;
  }
  return null;
}

/* ── Word complete banner ─────────────────────────────────────────────────── */
function showWordSuccess() {
  const { word } = practiceState;
  const vidWrap  = document.querySelector('.practice-vid-wrap');

  if (vidWrap) {
    const nextWord = getNextSupportedWord(practiceState.topicIdx, word);
    const banner = document.createElement('div');
    banner.className = 'practice-success';
    banner.innerHTML = `
      <div class="big-check">🎉</div>
      <div>Hoàn thành từ "<strong>${word}</strong>"!</div>
      <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;justify-content:center;">
        <button class="practice-manual-btn" id="practiceReplayBtn">🔄 Thử lại</button>
        <button class="practice-manual-btn" id="practiceNextWordBtn"
          style="background:linear-gradient(135deg,#16a34a,#15803d);box-shadow:0 2px 10px rgba(22,163,74,.35);"
          ${!nextWord ? 'disabled' : ''}>
          ⏭ Từ tiếp theo${nextWord ? `: "${nextWord}"` : ' (hết từ)'}
        </button>
      </div>`;
    vidWrap.appendChild(banner);

    banner.querySelector('#practiceReplayBtn').onclick = () => {
      banner.remove();
      practiceState.currentCharIdx = 0;
      refreshCharProgress();
      loadCharVideo(0);
      const nextBtn = document.getElementById('practiceNextBtn');
      if (nextBtn) { nextBtn.disabled = false; nextBtn.style.opacity = '1'; }
    };

    if (nextWord) {
      banner.querySelector('#practiceNextWordBtn').onclick = () => {
        startPracticeSession(practiceState.topicIdx, nextWord);
      };
    }
  }

  const nextBtn = document.getElementById('practiceNextBtn');
  if (nextBtn) { nextBtn.disabled = true; }

  const statusText = document.getElementById('statusText');
  if (statusText) statusText.innerHTML = `🎊 Tuyệt vời! Bạn đã hoàn thành từ "<strong>${word}</strong>"`;
}

/* ── Camera ───────────────────────────────────────────────────────────────── */
async function startCamera() {
  const videoEl     = document.getElementById('practiceCamera');
  const placeholder = document.getElementById('camPlaceholder');
  if (!videoEl) return;

  if (!handLandmarker) {
    try { await initHandLandmarker(); } catch (e) { console.warn('HandLandmarker init failed', e); }
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    practiceState.cameraStream = stream;
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';

    videoEl.onloadeddata = () => {
      handDetectRunning = true;
      detectHandLoop(videoEl);
      gruInferenceLoop();
    };
  } catch (err) {
    if (placeholder) {
      placeholder.innerHTML = `
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.5">
          <line x1="2" y1="2" x2="22" y2="22"/>
          <path d="M11 11a3 3 0 0 0 4.243 4.243M6.116 6.116
                   A7 7 0 0 0 17.657 17.657M21 21l-18-18"/>
        </svg>
        <span>Không truy cập được camera.<br><small>${err.message}</small></span>`;
    }
  }
}

function stopCamera() {
  handDetectRunning = false;
  practiceState.cameraStream?.getTracks().forEach(t => t.stop());
  practiceState.cameraStream = null;
}

/* ── Inference counter display ────────────────────────────────────────────── */
function setGruCounter(count, done, max = 30, modelLabel = 'GRU') {
  const wrap  = document.getElementById('gruCounter');
  const num   = document.getElementById('gruFrameCount');
  const maxEl = document.getElementById('gruFrameMax');
  const bar   = document.getElementById('gruCounterBar');
  const badge = document.getElementById('gruModelBadge');
  if (!wrap || !num || !bar) return;

  const pct = Math.round((count / max) * 100);
  num.textContent = count;
  if (maxEl) maxEl.textContent = `/${max}`;
  if (badge) { badge.textContent = modelLabel; badge.dataset.model = modelLabel.toLowerCase(); }

  if (count === 0 && !done) {
    bar.style.transition = 'none';
    bar.style.width = '0%';
    bar.offsetHeight;
    bar.style.transition = '';
  } else {
    bar.style.width = `${pct}%`;
  }

  wrap.dataset.state = done ? 'done' : count > 0 ? 'active' : 'idle';
}

/* ═══════════════════════════════════════════════════════════════════════════
   FIX BUG 1 — CAMERA FEEDBACK OVERLAY
   ═══════════════════════════════════════════════════════════════════════════ */
function showCameraFeedback(type, label, confidence) {
  const camWrap = document.querySelector('.practice-cam-wrap');
  if (!camWrap) return;

  camWrap.querySelector('.gru-feedback-overlay')?.remove();

  const isCorrect = type === 'correct';
  const overlay   = document.createElement('div');
  overlay.className = 'gru-feedback-overlay';

  Object.assign(overlay.style, {
    position:       'absolute',
    inset:          '0',
    zIndex:         '30',
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'center',
    justifyContent: 'center',
    gap:            '10px',
    background:     isCorrect
      ? 'rgba(34, 197, 94, 0.55)'
      : 'rgba(239, 68, 68, 0.50)',
    backdropFilter: 'blur(3px)',
    borderRadius:   '0',
    pointerEvents:  'none',
    animation:      'feedbackFadeIn .15s ease',
  });

  overlay.innerHTML = `
    <div style="font-size:3.2rem;line-height:1;filter:drop-shadow(0 2px 8px rgba(0,0,0,.4))">
      ${isCorrect ? '✅' : '🔄'}
    </div>
    <div style="
      color:#fff;
      font-weight:800;
      font-size:1.15rem;
      text-shadow:0 2px 10px rgba(0,0,0,.6);
      letter-spacing:-.01em;
    ">
      ${isCorrect ? 'Đúng rồi!' : 'Thử lại'}
    </div>
    <div style="
      color:rgba(255,255,255,.85);
      font-size:.82rem;
      font-family:'Space Mono',monospace;
      background:rgba(0,0,0,.25);
      padding:3px 12px;
      border-radius:20px;
    ">
      ${label} · ${confidence}%
    </div>`;

  camWrap.appendChild(overlay);
}

function hideCameraFeedback() {
  document.querySelector('.gru-feedback-overlay')?.remove();
}

if (!document.getElementById('gruFeedbackStyle')) {
  const style = document.createElement('style');
  style.id = 'gruFeedbackStyle';
  style.textContent = `
    @keyframes feedbackFadeIn {
      from { opacity: 0; transform: scale(.96); }
      to   { opacity: 1; transform: scale(1);   }
    }
  `;
  document.head.appendChild(style);
}

/* ═══════════════════════════════════════════════════════════════════════════
   HAND DETECTION LOOP
   ═══════════════════════════════════════════════════════════════════════════ */
let lastVideoTime      = -1;
let lastDetectTime     = 0;
let gruDisplaying      = false;
let latestLandmarks    = null;
let cachedCanvasW      = 0;
let cachedCanvasH      = 0;
let gruLoopGeneration  = 0;

const GRU_FRAME_INTERVAL  = 33;
const DETECT_INTERVAL     = 33;

/* ── MediaPipe loop ──────────────────────────────────────────────────────── */
function detectHandLoop(videoEl) {
  if (!handDetectRunning) return;

  const canvas = document.getElementById('handCanvas');
  if (!canvas || !handLandmarker || videoEl.readyState < 2) {
    requestAnimationFrame(() => detectHandLoop(videoEl));
    return;
  }

  const now = performance.now();

  if (now - lastDetectTime >= DETECT_INTERVAL && videoEl.currentTime !== lastVideoTime) {
    lastDetectTime = now;
    lastVideoTime  = videoEl.currentTime;

    const vw = videoEl.videoWidth  || videoEl.offsetWidth;
    const vh = videoEl.videoHeight || videoEl.offsetHeight;
    if (vw !== cachedCanvasW || vh !== cachedCanvasH) {
      canvas.width  = vw;
      canvas.height = vh;
      cachedCanvasW = vw;
      cachedCanvasH = vh;
    }

    const ctx     = canvas.getContext('2d');
    const results = handLandmarker.detectForVideo(videoEl, now);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks?.length) {
      drawHandMirrored(ctx, results.landmarks[0], canvas.width, canvas.height);
      latestLandmarks = results.landmarks[0];
      updateHandStatus(results.landmarks.length);
    } else {
      latestLandmarks = null;
      updateHandStatus(0);
    }
  }

  requestAnimationFrame(() => detectHandLoop(videoEl));
}

/* ── GRU inference loop ──────────────────────────────────────────────────── */
async function gruInferenceLoop() {
  const sleep  = ms => new Promise(r => setTimeout(r, ms));
  const myGen  = gruLoopGeneration;

  while (handDetectRunning && practiceState.active && gruLoopGeneration === myGen) {
    await sleep(GRU_FRAME_INTERVAL);

    if (gruDisplaying) continue;

    const gruResult = await pushKeypointsAndPredict(latestLandmarks);

    if (gruResult?.status === 'predicted' || gruResult?.status === 'low_confidence') {
      const matched = (gruResult.status === 'predicted' && gruResult.label === getCurrentGRULabel());

      // 🔒 Khóa thu thập frame ngay lập tức
      gruDisplaying = true;
      setGruCounter(30, true, 30, 'GRU');
      updateHandStatus(latestLandmarks ? 1 : 0, gruResult);

      if (matched) {
        // Đúng: hiện overlay xanh 1s, reset buffer + skip, rồi chuyển chữ tiếp
        showCameraFeedback('correct', gruResult.label, gruResult.confidence);
        await sleep(1000);
        hideCameraFeedback();
        // resetGRUBufferAndSkip(15) được gọi bên trong loadCharVideo (qua advanceChar)
        advanceChar();

      } else {
        // Sai: hiện overlay đỏ 1s, reset buffer + skip 15 frame (~500ms) để
        // counter thực sự về 0 và không nhảy lên 10-15 ngay khi tay còn trong frame.
        showCameraFeedback('wrong', gruResult.label, gruResult.confidence);
        await sleep(1000);
        hideCameraFeedback();
        resetGRUBufferAndSkip(15);      // ← thay resetGRUBuffer(), bắt buộc counter về 0
        setGruCounter(0, false, 30, 'GRU');
        gruDisplaying = false;          // 🔓 Mở khóa — skip frames xử lý bên trong gru_inference.js
      }

    } else if (gruResult?.status === 'buffering') {
      setGruCounter(gruResult.progress, false, 30, 'GRU');
    }
  }
}

/* ── Draw mirrored skeleton ───────────────────────────────────────────────── */
function drawHandMirrored(ctx, landmarks, W, H) {
  const px = (lm) => ({ x: (1 - lm.x) * W, y: lm.y * H });

  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = 'round';
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    const a = px(landmarks[start]);
    const b = px(landmarks[end]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  const tips = new Set([4, 8, 12, 16, 20]);
  for (let i = 0; i < landmarks.length; i++) {
    const { x, y } = px(landmarks[i]);
    const r = tips.has(i) ? 6 : 4;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle   = '#2563eb';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }
}

/* ── Update status text ───────────────────────────────────────────────────── */
function updateHandStatus(count, gruResult = null) {
  if (!practiceState.active) return;
  if (gruDisplaying && !gruResult) return;

  const statusText = document.getElementById('statusText');
  const statusDot  = document.getElementById('statusDot');
  const char = practiceState.chars?.[practiceState.currentCharIdx] ?? '';

  if (count === 0) {
    if (statusDot)  statusDot.className = 'status-indicator waiting';
    if (statusText) statusText.innerHTML = `Đưa tay vào khung hình để thực hiện chữ "<strong>${char}</strong>"`;
  } else if (gruResult?.status === 'predicted' || gruResult?.status === 'low_confidence') {
    const matched = gruResult.label === getCurrentGRULabel();
    const isLow   = gruResult.status === 'low_confidence';
    if (statusDot)  statusDot.className = `status-indicator ${matched ? 'correct' : 'waiting'}`;
    if (statusText) statusText.innerHTML = matched
      ? `✅ Nhận diện: <strong>${gruResult.label}</strong> (${gruResult.confidence}%) — Đúng rồi!`
      : isLow
        ? `🔍 Nhận diện: <strong>${gruResult.label}</strong> (${gruResult.confidence}%) — Chưa rõ, tiếp tục…`
        : `🔄 Nhận diện: <strong>${gruResult.label}</strong> (${gruResult.confidence}%) — Thử lại…`;
  } else {
    if (statusDot)  statusDot.className = 'status-indicator correct';
    if (statusText) statusText.innerHTML = `✋ Phát hiện ${count} tay — đang nhận diện chữ "<strong>${char}</strong>"…`;
  }
}

/* ── Exit practice session ────────────────────────────────────────────────── */
function exitPracticeSession() {
  gruLoopGeneration++;

  practiceState.active = false;
  gruDisplaying = false;
  stopCamera();
  hideCameraFeedback();

  cachedCanvasW = 0;
  cachedCanvasH = 0;

  document.getElementById('practiceView')?.remove();

  const featCard = document.querySelector('.featured-card');
  if (featCard) featCard.style.display = '';
}

/* ═══════════════════════════════════════════════════════════════════════════
   CAMERA CLEANUP ON NAVIGATION
   Dừng camera khi: (1) parent gửi message 'stopLearnCamera',
                    (2) iframe bị ẩn (visibilitychange),
                    (3) trang unload.
   ═══════════════════════════════════════════════════════════════════════════ */
function safeStopCamera() {
  if (practiceState.active) exitPracticeSession();
  else stopCamera();
}

window.addEventListener('message', (e) => {
  if (e.data?.action === 'stopLearnCamera') safeStopCamera();
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') safeStopCamera();
});

window.addEventListener('pagehide', safeStopCamera);

/* ═══════════════════════════════════════════════════════════════════════════
   BOOT
   ═══════════════════════════════════════════════════════════════════════════ */
renderSidebar();
renderFeaturedCard();
selectLesson(0, 0);