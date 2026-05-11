const API = "http://localhost:8000/api";

let currentSessionId = null;
let isLoading = false;

const DEFAULT_USER_IMG = "account.png";
const DEFAULT_BOT_IMG = "robot.png";

const avatarState = {
  user: { src: DEFAULT_USER_IMG },
  bot: { src: DEFAULT_BOT_IMG },
};

(function loadSavedAvatars() {
  try {
    const saved = localStorage.getItem("chatbot_avatars_v2");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.user?.src) avatarState.user.src = parsed.user.src;
      if (parsed.bot?.src) avatarState.bot.src = parsed.bot.src;
    }
  } catch (_) {}
})();

function saveAvatars() {
  try {
    localStorage.setItem("chatbot_avatars_v2", JSON.stringify(avatarState));
  } catch (_) {}
}

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send-btn");
const sessionList = document.getElementById("session-list");
const toastEl = document.getElementById("toast");

const modalOverlay = document.getElementById("avatar-modal-overlay");
const modalTitle = document.getElementById("avatar-modal-title");
const previewImg = document.getElementById("avatar-preview-img");
const fileInput = document.getElementById("avatar-file-input");

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 3500);
}

promptEl.addEventListener("input", () => {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

let editingRole = null;
let pendingSrc = null;

function openAvatarModal(role) {
  editingRole = role;
  pendingSrc = avatarState[role].src;
  modalTitle.textContent =
    role === "user" ? "Đổi ảnh người dùng" : "Đổi ảnh bot";
  previewImg.src = pendingSrc;
  modalOverlay.classList.add("open");
}

document.getElementById("btn-pick-file").addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    pendingSrc = e.target.result;
    previewImg.src = pendingSrc;
  };
  reader.readAsDataURL(file);
});

document.getElementById("btn-reset-avatar").addEventListener("click", () => {
  pendingSrc = editingRole === "user" ? DEFAULT_USER_IMG : DEFAULT_BOT_IMG;
  previewImg.src = pendingSrc;
});

document
  .getElementById("btn-modal-cancel")
  .addEventListener("click", closeModal);

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});

document.getElementById("btn-modal-save").addEventListener("click", () => {
  if (!editingRole) return;
  avatarState[editingRole].src = pendingSrc;
  saveAvatars();

  document
    .querySelectorAll(`.msg-avatar[data-role="${editingRole}"] img`)
    .forEach((img) => {
      img.src = pendingSrc;
    });

  closeModal();
});

function closeModal() {
  modalOverlay.classList.remove("open");
  editingRole = null;
  pendingSrc = null;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function hideWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function makeAvatar(role) {
  const wrap = document.createElement("div");
  wrap.className = "msg-avatar";
  wrap.dataset.role = role === "user" ? "user" : "bot";
  wrap.title = "Nhấn để đổi ảnh đại diện";

  const img = document.createElement("img");
  img.src = avatarState[role === "user" ? "user" : "bot"].src;
  img.alt = role === "user" ? "User" : "Bot";
  wrap.appendChild(img);

  wrap.addEventListener("click", () =>
    openAvatarModal(role === "user" ? "user" : "bot"),
  );
  return wrap;
}

// ── Render messages ──────────────────────────────────────────────────────────
function appendMessage(role, text) {
  hideWelcome();

  const row = document.createElement("div");
  row.className = `msg-row${role === "user" ? " user-row" : ""}`;

  const avatar = makeAvatar(role);

  const content = document.createElement("div");
  content.className = "msg-content";

  const sender = document.createElement("div");
  sender.className = "msg-sender";
  sender.textContent = role === "user" ? "Bạn" : "Bot";

  const msgText = document.createElement("div");
  msgText.className = "msg-text";
  msgText.textContent = text;

  content.appendChild(sender);
  content.appendChild(msgText);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendLoadingDots() {
  hideWelcome();

  const row = document.createElement("div");
  row.className = "msg-row";
  row.id = "loading-row";

  const avatar = makeAvatar("assistant");

  const content = document.createElement("div");
  content.className = "msg-content";

  const sender = document.createElement("div");
  sender.className = "msg-sender";
  sender.textContent = "Bot";

  const dots = document.createElement("div");
  dots.className = "dot-anim";
  dots.innerHTML = "<span></span><span></span><span></span>";

  content.appendChild(sender);
  content.appendChild(dots);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeLoading() {
  const el = document.getElementById("loading-row");
  if (el) el.remove();
}

// ── Session sidebar ──────────────────────────────────────────────────────────
async function loadSessionList() {
  try {
    const res = await fetch(`${API}/history`);
    if (!res.ok) return;
    const data = await res.json();
    renderSessionList(data.sessions || []);
  } catch (_) {
    /* server chưa chạy */
  }
}

function renderSessionList(sessions) {
  if (!sessions.length) {
    sessionList.innerHTML =
      '<div class="sidebar-empty">No conversations yet</div>';
    return;
  }
  sessionList.innerHTML = "";
  sessions.forEach((s) => {
    const btn = document.createElement("button");
    btn.className =
      "sess-btn" + (s.session_id === currentSessionId ? " active" : "");
    btn.textContent = s.title || s.session_id;
    btn.title = `${s.turn_count} lượt · ${s.updated_at}`;
    btn.addEventListener("click", () => loadSession(s.session_id));
    sessionList.appendChild(btn);
  });
}

async function loadSession(sessionId) {
  try {
    const res = await fetch(`${API}/history/${sessionId}`);
    if (!res.ok) throw new Error();
    const data = await res.json();

    currentSessionId = sessionId;
    messagesEl.innerHTML = "";

    const msgs = data.session.messages || [];
    if (!msgs.length) {
      showWelcome();
    } else {
      msgs.forEach((m) => appendMessage(m.role, m.content));
    }
    await loadSessionList();
  } catch (_) {
    showToast("Không thể tải phiên chat.");
  }
}

function showWelcome() {
  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>Sign Language Chatbot</h2>
      <p>Nhập tin nhắn để bắt đầu</p>
    </div>`;
}

document.getElementById("btn-new-chat").addEventListener("click", () => {
  currentSessionId = null;
  showWelcome();
  loadSessionList();
});

async function sendMessage() {
  if (isLoading) return;
  const text = promptEl.value.trim();
  if (!text) return;

  promptEl.value = "";
  promptEl.style.height = "auto";
  appendMessage("user", text);

  isLoading = true;
  sendBtn.disabled = true;
  appendLoadingDots();

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId || undefined,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    removeLoading();
    if (!currentSessionId) currentSessionId = data.session_id;
    appendMessage("assistant", data.response);
    await loadSessionList();
  } catch (_) {
    removeLoading();
    showToast("Lỗi kết nối. Kiểm tra FastAPI đang chạy tại port 8000.");
  }

  isLoading = false;
  sendBtn.disabled = false;
}

loadSessionList();
