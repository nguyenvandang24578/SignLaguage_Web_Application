const API = "http://localhost:8000/api";

let currentSessionId = null;
let isLoading = false;
let currentTaskId = null;
let pollingInterval = null;

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send-btn");
const cancelBtn = document.getElementById("cancel-btn");
const sessionList = document.getElementById("session-list");
const toastEl = document.getElementById("toast");

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
cancelBtn.addEventListener("click", cancelCurrentTask);

// ── Helpers ───────────────────────────────────────────────────────────────────
function hideWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function makeAvatar(role) {
  const wrap = document.createElement("div");
  wrap.className = "msg-avatar";
  wrap.dataset.role = role === "user" ? "user" : "bot";

  const img = document.createElement("img");
  img.src = role === "user" ? "account.png" : "robot.png";
  img.alt = role === "user" ? "User" : "Bot";
  wrap.appendChild(img);

  return wrap;
}

// ── Render messages ───────────────────────────────────────────────────────────
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

function appendLoadingIndicator() {
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

  const indicator = document.createElement("div");
  indicator.className = "msg-indicator";
  indicator.innerHTML = `
    <div class="dot-anim"><span></span><span></span><span></span></div>
  `;

  content.appendChild(sender);
  content.appendChild(indicator);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  // Show cancel button in input area, hide send button
  setLoadingState(true);
}

function removeLoading() {
  const el = document.getElementById("loading-row");
  if (el) el.remove();
  setLoadingState(false);
}

function setLoadingState(isLoading) {
  if (isLoading) {
    sendBtn.style.display = "none";
    cancelBtn.style.display = "flex";
  } else {
    sendBtn.style.display = "flex";
    cancelBtn.style.display = "none";
  }
}

function replaceLoadingWithMessage(text, toolsUsed = []) {
  const row = document.getElementById("loading-row");
  if (!row) {
    appendMessage("assistant", text);
    return;
  }

  row.removeAttribute("id");
  const content = row.querySelector(".msg-content");
  if (!content) {
    appendMessage("assistant", text);
    return;
  }

  const indicator = content.querySelector(".msg-indicator");
  if (indicator) indicator.remove();

  if (Array.isArray(toolsUsed) && toolsUsed.length > 0) {
    const tools = document.createElement("div");
    tools.className = "msg-tools";
    tools.textContent = `Tool: ${toolsUsed.join(", ")}`;
    content.appendChild(tools);
  }

  const msgText = document.createElement("div");
  msgText.className = "msg-text";
  msgText.textContent = text;
  content.appendChild(msgText);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  setLoadingState(false);
}

// ── Session sidebar ───────────────────────────────────────────────────────────
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
  // Lấy danh sách session_id hiện tại trên DOM
  const existingItems = new Map();
  sessionList.querySelectorAll(".sess-item[data-sid]").forEach((el) => {
    existingItems.set(el.dataset.sid, el);
  });

  if (!sessions.length) {
    sessionList.innerHTML =
      '<div class="sidebar-empty">No conversations yet</div>';
    return;
  }

  // Xóa empty placeholder nếu có
  const empty = sessionList.querySelector(".sidebar-empty");
  if (empty) empty.remove();

  const newIds = new Set(sessions.map((s) => s.session_id));

  // Xóa các item không còn tồn tại
  existingItems.forEach((el, sid) => {
    if (!newIds.has(sid)) el.remove();
  });

  // Update hoặc tạo mới từng item — KHÔNG xóa toàn bộ innerHTML
  sessions.forEach((s, index) => {
    let item = existingItems.get(s.session_id);

    if (item) {
      // Update item đã có: chỉ update text và class, không tạo lại listener
      item.className =
        "sess-item" + (s.session_id === currentSessionId ? " active" : "");
      const btn = item.querySelector(".sess-btn");
      if (btn) {
        btn.textContent = s.title || s.session_id;
        btn.title = `${s.turn_count} lượt · ${s.updated_at}`;
      }
    } else {
      // Tạo item mới
      item = document.createElement("div");
      item.className =
        "sess-item" + (s.session_id === currentSessionId ? " active" : "");
      item.dataset.sid = s.session_id;

      const btn = document.createElement("button");
      btn.className = "sess-btn";
      btn.textContent = s.title || s.session_id;
      btn.title = `${s.turn_count} lượt · ${s.updated_at}`;
      btn.addEventListener("click", () => loadSession(s.session_id));

      const delBtn = document.createElement("button");
      delBtn.className = "sess-del-btn";
      delBtn.title = "Xoá cuộc trò chuyện này";
      delBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>`;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        confirmDelete(s.session_id, s.title || s.session_id);
      });

      item.appendChild(btn);
      item.appendChild(delBtn);
      sessionList.appendChild(item);
    }

    // Đảm bảo thứ tự đúng
    if (sessionList.children[index] !== item) {
      sessionList.insertBefore(item, sessionList.children[index] || null);
    }
  });
}

function confirmDelete(sessionId, title) {
  const modal = document.getElementById("confirm-modal");
  document.getElementById("confirm-modal-title").textContent =
    `Xoá "${title.length > 30 ? title.slice(0, 30) + "…" : title}"?`;
  modal.classList.add("open");

  // Clone để xóa sạch listener cũ, tránh fire nhiều lần
  const yesBtn = document.getElementById("btn-confirm-yes");
  const noBtn = document.getElementById("btn-confirm-no");
  const newYes = yesBtn.cloneNode(true);
  const newNo = noBtn.cloneNode(true);
  yesBtn.replaceWith(newYes);
  noBtn.replaceWith(newNo);

  newYes.addEventListener("click", async () => {
    modal.classList.remove("open");
    await deleteSession(sessionId);
  });
  newNo.addEventListener("click", () => {
    modal.classList.remove("open");
  });
}

async function deleteSession(sessionId) {
  try {
    const res = await fetch(`${API}/history/${sessionId}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error();
    // Chỉ reset màn hình nếu đúng session đang xem bị xóa
    if (currentSessionId && currentSessionId === sessionId) {
      currentSessionId = null;
      showWelcome();
    }
    showToast("Đã xoá cuộc trò chuyện.");
    await loadSessionList();
  } catch (_) {
    showToast("Không thể xoá. Kiểm tra server.");
  }
}

async function loadSession(sessionId) {
  console.log("loadSession called:", sessionId, "current:", currentSessionId);
  console.trace();
  // FIX: Không load lại nếu đang xem đúng session đó
  if (sessionId === currentSessionId) return;

  // Stop any ongoing polling when switching conversations
  stopPolling();
  isLoading = false;
  sendBtn.disabled = false;
  removeLoading();

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
  console.trace("⚠️ showWelcome called — stack:");
  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>VNSignMate Assistant</h2>
      <p>Nhập tin nhắn để bắt đầu</p>
    </div>`;
}

document.getElementById("btn-new-chat").addEventListener("click", () => {
  // Stop any ongoing polling when starting a new chat
  stopPolling();
  isLoading = false;
  sendBtn.disabled = false;
  removeLoading();

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

  const currentSessionIdBefore = currentSessionId;
  isLoading = true;
  sendBtn.disabled = true;
  appendLoadingIndicator();

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

    // Backend now returns task_id immediately
    currentTaskId = data.task_id;
    if (!currentSessionId) currentSessionId = data.session_id;

    // Start polling for task completion
    startPolling(currentTaskId);

    // Chỉ reload sidebar nếu đây là tin nhắn đầu tiên (session mới có title)
    // Tránh re-render sidebar không cần thiết gây văng màn hình
    const isFirstMessage = !currentSessionIdBefore;
    if (isFirstMessage) await loadSessionList();
  } catch (_) {
    removeLoading();
    showToast("Lỗi kết nối. Kiểm tra FastAPI đang chạy tại port 8000.");
    isLoading = false;
    sendBtn.disabled = false;
  }
}

function startPolling(taskId) {
  // Clear any existing polling
  if (pollingInterval) {
    clearInterval(pollingInterval);
  }

  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API}/chat/status/${taskId}`);
      if (!res.ok) {
        if (res.status === 404) {
          stopPolling();
          removeLoading();
          showToast("Task not found");
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();

      // Update loading indicator with progress
      updateLoadingProgress(data.progress, data.current_step);

      if (data.status === "completed") {
        stopPolling();
        const toolsUsed = Array.isArray(data.result?.tools_used) ? data.result.tools_used : [];
        replaceLoadingWithMessage(data.result?.response || "", toolsUsed);
        isLoading = false;
        sendBtn.disabled = false;
      } else if (data.status === "failed") {
        stopPolling();
        removeLoading();
        showToast(`Lỗi: ${data.error || "Unknown error"}`);
        isLoading = false;
        sendBtn.disabled = false;
      } else if (data.status === "cancelled") {
        stopPolling();
        const toolsUsed = Array.isArray(data.result?.tools_used) ? data.result.tools_used : [];
        const response = data.result?.response || "";
        const isPartial = data.result?.partial === true;
        
        if (response) {
          // Show partial response with indicator
          replaceLoadingWithMessage(response + (isPartial ? " ⚠️ (partial)" : ""), toolsUsed);
        } else {
          removeLoading();
        }
        showToast(isPartial ? "Generation cancelled - partial response saved" : "Generation cancelled");
        isLoading = false;
        sendBtn.disabled = false;
      }
    } catch (err) {
      console.error("Polling error:", err);
      // Don't stop polling on network errors, retry
    }
  }, 500); // Poll every 500ms
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
  currentTaskId = null;
}

async function cancelCurrentTask() {
  if (!currentTaskId) return;
  
  try {
    const res = await fetch(`${API}/chat/cancel/${currentTaskId}`, {
      method: "POST",
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    
    // The polling will pick up the cancelled status
    showToast("Generation cancelled");
  } catch (err) {
    console.error("Cancel error:", err);
    showToast("Failed to cancel");
  }
}

// Handle Escape key to cancel generation
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isLoading && currentTaskId) {
    cancelCurrentTask();
  }
});

function updateLoadingProgress(progress, step) {
  const row = document.getElementById("loading-row");
  if (!row) return;

  const indicator = row.querySelector(".msg-indicator");
  if (indicator) {
    // Update the progress text while preserving the cancel button
    let progressEl = indicator.querySelector(".loading-progress");
    if (!progressEl) {
      progressEl = document.createElement("div");
      progressEl.className = "loading-progress";
      indicator.appendChild(progressEl);
    }
    progressEl.textContent = `${Math.round(progress * 100)}% - ${step}`;
  }
}

loadSessionList();
