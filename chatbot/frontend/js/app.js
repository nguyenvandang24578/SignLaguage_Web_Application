const API = "http://localhost:8000/api";

let currentSessionId = null;
let isLoading = false;

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send-btn");
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

function hideWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function makeAvatar(role) {
  const wrap = document.createElement("div");
  wrap.className = "msg-avatar";
  wrap.dataset.role = role === "user" ? "user" : "bot";

  const img = document.createElement("img");
  img.src = role === "user" ? "assets/account.png" : "assets/robot.png";
  img.alt = role === "user" ? "User" : "Bot";
  wrap.appendChild(img);

  return wrap;
}

function appendMessage(role, text, toolsUsed = [], elapsed = null) {
  hideWelcome();

  const row = document.createElement("div");
  row.className = `msg-row${role === "user" ? " user-row" : ""}`;

  const avatar = makeAvatar(role);

  const content = document.createElement("div");
  content.className = "msg-content";

  const sender = document.createElement("div");
  sender.className = "msg-sender";

  if (role === "user") {
    sender.textContent = "Bạn";
  } else {
    sender.textContent = "Bot";
    if (elapsed !== null) {
      const timeSpan = document.createElement("span");
      timeSpan.className = "msg-time";
      timeSpan.textContent = `${elapsed.toFixed(1)}s`;
      sender.appendChild(timeSpan);
    }
  }

  const msgText = document.createElement("div");
  msgText.className = "msg-text";
  msgText.textContent = text;

  content.appendChild(sender);
  content.appendChild(msgText);

  if (role !== "user" && Array.isArray(toolsUsed) && toolsUsed.length > 0) {
    const tools = document.createElement("div");
    tools.className = "msg-tools";
    tools.textContent = `${toolsUsed.join(", ")}`;
    content.appendChild(tools);
  }

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

  sendBtn.disabled = true;
}

function removeLoading() {
  const el = document.getElementById("loading-row");
  if (el) el.remove();
  sendBtn.disabled = false;
}

async function loadSessionList() {
  try {
    const res = await fetch(`${API}/history`);
    if (!res.ok) return;
    const data = await res.json();
    renderSessionList(data.sessions || []);
  } catch (_) {}
}

function renderSessionList(sessions) {
  const existingItems = new Map();
  sessionList.querySelectorAll(".sess-item[data-sid]").forEach((el) => {
    existingItems.set(el.dataset.sid, el);
  });

  if (!sessions.length) {
    sessionList.innerHTML =
      '<div class="sidebar-empty">No conversations yet</div>';
    return;
  }

  const empty = sessionList.querySelector(".sidebar-empty");
  if (empty) empty.remove();

  const newIds = new Set(sessions.map((s) => s.session_id));

  existingItems.forEach((el, sid) => {
    if (!newIds.has(sid)) el.remove();
  });

  sessions.forEach((s, index) => {
    let item = existingItems.get(s.session_id);

    if (item) {
      item.className =
        "sess-item" + (s.session_id === currentSessionId ? " active" : "");
      const btn = item.querySelector(".sess-btn");
      if (btn) {
        btn.textContent = s.title || s.session_id;
        btn.title = `${s.turn_count} lượt · ${s.updated_at}`;
      }
    } else {
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
  if (sessionId === currentSessionId) return;
  isLoading = false;
  if (currentAbortController) {
    currentAbortController.abort();
  }
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
  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>VNSignMate Assistant</h2>
      <p>Nhập tin nhắn để bắt đầu</p>
    </div>`;
}

document.getElementById("btn-new-chat").addEventListener("click", () => {
  isLoading = false;
  if (currentAbortController) {
    currentAbortController.abort();
  }
  removeLoading();
  currentSessionId = null;
  showWelcome();
  loadSessionList();
});

let currentAbortController = null;

// ─── Streaming message helpers ───────────────────────────────

function createBotMessageContainer() {
  hideWelcome();

  const row = document.createElement("div");
  row.className = "msg-row";
  row.id = "streaming-row";

  const avatar = makeAvatar("assistant");

  const content = document.createElement("div");
  content.className = "msg-content";

  const sender = document.createElement("div");
  sender.className = "msg-sender";
  sender.textContent = "Bot";
  content.appendChild(sender);

  const msgText = document.createElement("div");
  msgText.className = "msg-text";
  msgText.id = "streaming-text";
  content.appendChild(msgText);

  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  sendBtn.disabled = true;
  return { row, msgText };
}

function finalizeBotMessage(sessionId, toolsUsed, elapsed) {
  const row = document.getElementById("streaming-row");
  if (!row) return;

  row.removeAttribute("id");
  const textEl = row.querySelector("#streaming-text");
  if (textEl) textEl.removeAttribute("id");

  // Add elapsed time
  const sender = row.querySelector(".msg-sender");
  if (sender && elapsed !== null) {
    const timeSpan = document.createElement("span");
    timeSpan.className = "msg-time";
    timeSpan.textContent = `${elapsed.toFixed(1)}s`;
    sender.appendChild(timeSpan);
  }

  // Add tools used
  if (Array.isArray(toolsUsed) && toolsUsed.length > 0) {
    const content = row.querySelector(".msg-content");
    const tools = document.createElement("div");
    tools.className = "msg-tools";
    tools.textContent = `${toolsUsed.join(", ")}`;
    content.appendChild(tools);
  }

  if (!currentSessionId && sessionId) {
    currentSessionId = sessionId;
  }

  sendBtn.disabled = false;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateStreamingText(text) {
  const el = document.getElementById("streaming-text");
  if (el) {
    el.textContent = text;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

async function sendMessage() {
  if (isLoading) return;
  const text = promptEl.value.trim();
  if (!text) return;

  promptEl.value = "";
  promptEl.style.height = "auto";
  appendMessage("user", text);

  const currentSessionIdBefore = currentSessionId;
  const startTime = performance.now();
  isLoading = true;

  // Replace loading dots with a real streaming container
  appendLoadingIndicator();

  // ── Cancel / Abort controller ──────────────────────────
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();
  const controller = currentAbortController;

  const sendParent = sendBtn.parentElement;
  let cancelBtn = document.getElementById("cancel-stream-btn");
  if (!cancelBtn) {
    cancelBtn = document.createElement("button");
    cancelBtn.id = "cancel-stream-btn";
    cancelBtn.className = "cancel-btn";
    cancelBtn.title = "Dừng";
    cancelBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
    // Single persistent listener — always aborts the *current* controller
    cancelBtn.addEventListener("click", () => {
      if (currentAbortController) currentAbortController.abort();
    });
    sendParent.appendChild(cancelBtn);
  }
  cancelBtn.style.display = "flex";
  sendBtn.style.display = "none";

  try {
    const res = await fetch(`${API}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId || undefined,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    // Remove loading indicator, create real streaming container
    removeLoading();
    const { msgText } = createBotMessageContainer();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullResponse = "";
    let tempSessionId = null;
    let tempTools = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events: "data: {...}\n\n"
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          switch (event.type) {
            case "token":
              fullResponse += event.content;
              updateStreamingText(fullResponse);
              break;
            case "info":
              // Optionally: show a small status indicator
              break;
            case "done":
              tempSessionId = event.session_id || null;
              tempTools = event.tools_used || [];
              break;
            case "error":
              throw new Error(event.content);
          }
        } catch (parseErr) {
          if (parseErr.message !== "Unexpected end of JSON input") {
            console.warn("SSE parse error:", parseErr);
          }
        }
      }
    }

    const elapsed = (performance.now() - startTime) / 1000;

    finalizeBotMessage(tempSessionId, tempTools, elapsed);

    const isFirstMessage = !currentSessionIdBefore;
    if (isFirstMessage && tempSessionId) {
      currentSessionId = tempSessionId;
      await loadSessionList();
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // User cancelled — keep whatever we streamed so far
      const textEl = document.getElementById("streaming-text");
      if (textEl && textEl.textContent.trim()) {
        // If we have partial content, keep it
        finalizeBotMessage(null, [], (performance.now() - startTime) / 1000);
      } else {
        // Replace loading dots with a cancelled message
        removeLoading();
        const { msgText } = createBotMessageContainer();
        msgText.textContent = "\u0110\xe3 d\u1eebng";
        msgText.style.opacity = "0.5";
        finalizeBotMessage(null, [], (performance.now() - startTime) / 1000);
      }
    } else {
      removeLoading();
      // If streaming container already exists, reuse it
      const existingText = document.getElementById("streaming-text");
      if (existingText) {
        existingText.textContent += "\n\n[L\u1ed7i: " + err.message + "]";
        existingText.style.color = "#e74c3c";
        finalizeBotMessage(null, [], (performance.now() - startTime) / 1000);
      } else {
        const { msgText } = createBotMessageContainer();
        msgText.textContent = "L\u1ed7i: " + err.message;
        msgText.style.color = "#e74c3c";
        finalizeBotMessage(null, [], (performance.now() - startTime) / 1000);
      }
    }
  } finally {
    // Clean up abort controller
    if (currentAbortController === controller) {
      currentAbortController = null;
    }
    // Restore send button, hide cancel button
    const cancelBtnEl = document.getElementById("cancel-stream-btn");
    if (cancelBtnEl) cancelBtnEl.style.display = "none";
    sendBtn.style.display = "flex";
    isLoading = false;
  }
}

loadSessionList();
