// ============================================================================
// quiz.js — Sign Language Quiz (THIN CLIENT)
// ============================================================================
// Architecture:
//   • Browser: capture webcam → JPEG → send WS → receive state → update UI
//   • Server: MediaPipe Holistic + STA-GCN (native Python, fast)
//   • No skeleton overlay — clean camera view, prevents mirror/scale mismatch
//
// Anti-lag design (critical):
//   1. Send-Then-Wait: only 1 frame in network at a time. Backlog impossible.
//   2. Lower resolution (480×360) + JPEG q=0.5 → smaller payload, faster server.
//   3. No browser-side MediaPipe → CPU free for video decode + UI.
//   4. Frame drop on stale: if server response arrives "late" (>200ms after
//      send), skip waiting and send a fresh frame.
// ============================================================================

// ============================================================================
// 1. DATA + TOPIC CLASSIFICATION
// ============================================================================
const quizData = [
    { id: 0,  correctLabel: "ĂN",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/an.mp4",           topic: "action" },
    { id: 1,  correctLabel: "BẠN",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ban.mp4",           topic: "emotion" },
    { id: 2,  correctLabel: "BÀN CHÂN",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/banchan.mp4",       topic: "body" },
    { id: 3,  correctLabel: "BÉ GÁI",        videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/begai.mp4",         topic: "emotion" },
    { id: 4,  correctLabel: "BÉO",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/beo.mp4",           topic: "body" },
    { id: 5,  correctLabel: "BÉ TRAI",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/betrai.mp4",        topic: "emotion" },
    { id: 6,  correctLabel: "BỐ",            videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/bo.mp4",            topic: "emotion" },
    { id: 7,  correctLabel: "CAO",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/cao.mp4",           topic: "body" },
    { id: 8,  correctLabel: "CHẢI ĐẦU",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/chaidau.mp4",       topic: "hygiene" },
    { id: 9,  correctLabel: "CHẠY",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/chay.mp4",          topic: "action" },
    { id: 10, correctLabel: "CỔ",            videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/co.mp4",            topic: "body" },
    { id: 11, correctLabel: "CƯỜI",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/cuoi.mp4",          topic: "emotion" },
    { id: 12, correctLabel: "ĐÁNH RĂNG",     videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/danhrang.mp4",      topic: "hygiene" },
    { id: 13, correctLabel: "ĐẦU",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/dau.mp4",           topic: "body" },
    { id: 14, correctLabel: "ĐI",            videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/di.mp4",            topic: "action" },
    { id: 15, correctLabel: "ĐI VỆ SINH",    videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/divesinh.mp4",      topic: "hygiene" },
    { id: 16, correctLabel: "ĐÔI DÉP",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/doidep.mp4",       topic: "object" },
    { id: 17, correctLabel: "ĐỨNG",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/dung.mp4",          topic: "action" },
    { id: 18, correctLabel: "GĂNG TAY",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/gangtay.mp4",       topic: "object" },
    { id: 19, correctLabel: "GẦY",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/gay.mp4",           topic: "body" },
    // id 20 = NOTHING → excluded
    { id: 21, correctLabel: "BAO NHIÊU",     videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/baonhieu.mp4",      topic: "emotion" },
    { id: 22, correctLabel: "BÍT TẤT",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/bittat.mp4",        topic: "object" },
    { id: 23, correctLabel: "CẶP TÓC",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/captoc.mp4",        topic: "object" },
    { id: 24, correctLabel: "CHÀO",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/chao.mp4",          topic: "emotion" },
    { id: 25, correctLabel: "EM TRAI",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/emtrai.mp4",        topic: "emotion" },
    { id: 26, correctLabel: "GỘI ĐẦU",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/goidau.mp4",        topic: "hygiene" },
    { id: 27, correctLabel: "KHĂN MẶT",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/khanmat.mp4",       topic: "object" },
    { id: 28, correctLabel: "KHÓC",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/khoc.mp4",          topic: "emotion" },
    { id: 29, correctLabel: "KHỎE MẠNH",     videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/khoemanh.mp4",      topic: "emotion" },
    { id: 30, correctLabel: "KÍNH",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/kinh.mp4",          topic: "object" },
    { id: 31, correctLabel: "LƯỢC",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/luoc.mp4",          topic: "object" },
    { id: 32, correctLabel: "MÁ",            videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ma.mp4",            topic: "body" },
    { id: 33, correctLabel: "MẶC QUẦN ÁO",   videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/macquanao.mp4",     topic: "action" },
    { id: 34, correctLabel: "MẮT",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/mat.mp4",           topic: "body" },
    { id: 35, correctLabel: "MỆT MỎI",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/metmoi.mp4",        topic: "emotion" },
    { id: 36, correctLabel: "MIỆNG",         videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/mieng.mp4",         topic: "body" },
    { id: 37, correctLabel: "MŨI",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/mui.mp4",           topic: "body" },
    { id: 38, correctLabel: "MŨ LƯỠI TRAI",  videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/muluoitrai.mp4",    topic: "object" },
    { id: 39, correctLabel: "NẰM",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/nam.mp4",           topic: "action" },
    { id: 40, correctLabel: "NGỒI",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ngoi.mp4",          topic: "action" },
    { id: 41, correctLabel: "NGỦ",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ngu.mp4",           topic: "action" },
    { id: 42, correctLabel: "NHẢY",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/nhay.mp4",          topic: "action" },
    { id: 43, correctLabel: "NIỀM VUI",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/niemvui.mp4",       topic: "emotion" },
    { id: 44, correctLabel: "NÓN",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/non.mp4",           topic: "object" },
    { id: 45, correctLabel: "RỬA CHÂN",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ruachan.mp4",       topic: "hygiene" },
    { id: 46, correctLabel: "RỬA MẶT",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ruamat.mp4",        topic: "hygiene" },
    { id: 47, correctLabel: "RỬA TAY",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/ruatay.mp4",        topic: "hygiene" },
    { id: 48, correctLabel: "SẠCH",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/sach.mp4",          topic: "object" },
    { id: 49, correctLabel: "SỨC KHỎE",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/suckhoe.mp4",       topic: "emotion" }
];

const validQuizData = quizData.filter(item => item.videoUrl !== "" && item.id !== 20);
const allVocabs = validQuizData.map(item => item.correctLabel);

// ============================================================================
// 2. CONFIG
// ============================================================================
const CAMERA_WIDTH = 480;        // Lower res = faster server MediaPipe
const CAMERA_HEIGHT = 360;
const JPEG_QUALITY = 0.5;        // Quality vs size tradeoff (keypoints don't need pretty)
const FRAME_STALENESS_MS = 200;  // Drop response if older than this
const FRAME_MAX_INTERVAL = 33;   // Min time between send attempts (~30fps cap)
const WS_URL = `ws://${window.location.hostname || '127.0.0.1'}:8000/ws/practice`;

// ============================================================================
// 3. STATE
// ============================================================================
let historyStack = ['screen-main-menu'];

// Quiz
let currentQuestionIndex = 0;
let score = 0;
let wrongAnswers = [];
let currentQuizPool = [];
let quizSettings = { topic: 'all', count: 10, timer: 0 };
let timerInterval = null;
let timerSecondsLeft = 0;

// Practice / Challenge
let practiceMode = 'specific';
let challengeSettings = { count: 5 };
let challengePool = [];
let challengeIndex = 0;
let challengeScore = 0;
let challengeWrong = [];

// Camera + WS
let practiceWS = null;
let practiceStream = null;
let practiceVideoEl = null;
let practiceRAF = null;
let captureCanvas = null;          // Offscreen canvas for JPEG capture
let captureCtx = null;
let inflightFrameTs = 0;           // Timestamp of frame currently in network (0 = none)
let lastSendTime = 0;

// ============================================================================
// 4. CHIP SELECTION
// ============================================================================
function initChips() {
    document.querySelectorAll('.chip-group').forEach(group => {
        group.addEventListener('click', e => {
            const chip = e.target.closest('.chip');
            if (!chip) return;
            group.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
        });
    });
}

function getSelectedChipValue(groupId, attr) {
    const group = document.getElementById(groupId);
    if (!group) return null;
    const active = group.querySelector('.chip.active');
    return active ? active.dataset[attr] : null;
}

window.launchQuiz = function () {
    quizSettings.topic = getSelectedChipValue('topic-chips', 'topic') || 'all';
    quizSettings.count = parseInt(getSelectedChipValue('count-chips', 'count') || '10');
    quizSettings.timer = parseInt(getSelectedChipValue('timer-chips', 'timer') || '0');
    switchScreen('screen-quiz');
};

window.launchChallenge = function () {
    challengeSettings.count = parseInt(getSelectedChipValue('challenge-count-chips', 'count') || '5');
    challengePool = shuffleArray([...validQuizData]).slice(0, challengeSettings.count);
    challengeIndex = 0;
    challengeScore = 0;
    challengeWrong = [];
    practiceMode = 'challenge';
    document.getElementById('target-word').innerText = challengePool[0].correctLabel;
    switchScreen('screen-practice');
};

// ============================================================================
// 5. SCREEN NAVIGATION
// ============================================================================
function switchScreen(screenId, isBack = false) {
    clearQuizTimer();
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(screenId);
    if (target) target.classList.add('active');

    if (!isBack && historyStack[historyStack.length - 1] !== screenId) {
        historyStack.push(screenId);
    }

    const backBtn = document.getElementById('back-btn');
    const titleObj = document.getElementById('header-title');
    const scoreDisplay = document.getElementById('score-display');
    const progressBar = document.getElementById('quiz-progress-bar');

    scoreDisplay.style.display = 'none';
    progressBar.style.display = 'none';

    if (screenId !== 'screen-practice') stopPracticeSession();

    if (screenId === 'screen-main-menu') {
        backBtn.style.display = 'none';
        titleObj.innerText = "Trạm Kiểm Tra Năng Lực";
        return;
    }

    backBtn.style.display = 'flex';
    switch (screenId) {
        case 'screen-quiz-setup':       titleObj.innerText = "Cài Đặt Quiz"; break;
        case 'screen-practice-menu':    titleObj.innerText = "Giáo Án Luyện Tập"; break;
        case 'screen-challenge-setup':  titleObj.innerText = "Thử Thách Ngẫu Nhiên"; break;
        case 'screen-quiz':
            titleObj.innerText = "Giải Mã Ký Hiệu";
            scoreDisplay.style.display = 'flex';
            progressBar.style.display = 'block';
            startNewQuizSession();
            break;
        case 'screen-results':          titleObj.innerText = "Kết Quả"; break;
        case 'screen-practice':
            titleObj.innerText = practiceMode === 'challenge'
                ? `Thử Thách ${challengeIndex + 1}/${challengePool.length}`
                : "AI Tracking";
            initPracticeSession();
            break;
    }
}
window.switchScreen = switchScreen;

document.getElementById('back-btn').addEventListener('click', () => {
    if (historyStack.length > 1) {
        historyStack.pop();
        switchScreen(historyStack[historyStack.length - 1], true);
    }
});

// ============================================================================
// 6. VIDEO QUIZ ENGINE
// ============================================================================
function startNewQuizSession() {
    currentQuestionIndex = 0;
    score = 0;
    wrongAnswers = [];
    updateScoreDisplay();

    let pool = validQuizData;
    if (quizSettings.topic !== 'all') {
        pool = validQuizData.filter(q => q.topic === quizSettings.topic);
    }
    if (pool.length < 4) {
        document.getElementById('video-container').innerHTML =
            '<div class="video-placeholder"><i class="fas fa-exclamation-triangle"></i><p>Chủ đề này cần ít nhất 4 từ. Hãy chọn chủ đề khác.</p></div>';
        return;
    }

    currentQuizPool = shuffleArray([...pool]).slice(0, Math.min(quizSettings.count, pool.length));
    document.getElementById('score-total').textContent = currentQuizPool.length;
    loadNextQuestion();
}

function loadNextQuestion() {
    clearQuizTimer();
    document.getElementById('next-btn-container').style.display = 'none';
    if (currentQuestionIndex >= currentQuizPool.length) { showQuizResults(); return; }

    const q = currentQuizPool[currentQuestionIndex];
    document.getElementById('question-counter').textContent = `Câu ${currentQuestionIndex + 1} / ${currentQuizPool.length}`;
    document.getElementById('progress-fill').style.width = `${(currentQuestionIndex / currentQuizPool.length) * 100}%`;
    document.getElementById('score-progress').textContent = currentQuestionIndex;

    document.getElementById('video-container').innerHTML = `
        <video autoplay loop muted playsinline>
            <source src="${q.videoUrl}" type="video/mp4">
        </video>`;

    let options = [q.correctLabel];
    const others = shuffleArray(allVocabs.filter(v => v !== q.correctLabel));
    for (let i = 0; i < 3; i++) options.push(others[i]);
    options = shuffleArray(options);

    const grid = document.getElementById('options-grid');
    grid.innerHTML = '';
    options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = opt;
        btn.addEventListener('click', () => checkAnswer(btn, opt, q.correctLabel));
        grid.appendChild(btn);
    });

    if (quizSettings.timer > 0) startQuizTimer(quizSettings.timer, q.correctLabel);
}

function checkAnswer(clickedBtn, selected, correct) {
    clearQuizTimer();
    const allBtns = document.querySelectorAll('#options-grid .option-btn');
    allBtns.forEach(b => { b.style.pointerEvents = 'none'; b.classList.add('answered'); });

    if (selected === correct) {
        score++;
        clickedBtn.classList.add('correct-answer');
    } else {
        clickedBtn.classList.add('wrong-answer');
        wrongAnswers.push({
            question: currentQuizPool[currentQuestionIndex].correctLabel,
            yourAnswer: selected, correctAnswer: correct
        });
        allBtns.forEach(b => { if (b.textContent === correct) b.classList.add('correct-answer'); });
    }
    updateScoreDisplay();
    document.getElementById('next-btn-container').style.display = 'block';
    document.getElementById('next-question-btn').onclick = () => { currentQuestionIndex++; loadNextQuestion(); };
}

function handleTimeout(correct) {
    wrongAnswers.push({
        question: currentQuizPool[currentQuestionIndex].correctLabel,
        yourAnswer: "(Hết giờ)", correctAnswer: correct
    });
    const allBtns = document.querySelectorAll('#options-grid .option-btn');
    allBtns.forEach(b => {
        b.style.pointerEvents = 'none';
        b.classList.add('answered');
        if (b.textContent === correct) b.classList.add('correct-answer');
    });
    updateScoreDisplay();
    document.getElementById('next-btn-container').style.display = 'block';
    document.getElementById('next-question-btn').onclick = () => { currentQuestionIndex++; loadNextQuestion(); };
}

function startQuizTimer(seconds, correct) {
    timerSecondsLeft = seconds;
    const display = document.getElementById('timer-display');
    const secEl = document.getElementById('timer-seconds');
    display.style.display = 'flex';
    display.classList.remove('urgent');
    secEl.textContent = timerSecondsLeft;
    timerInterval = setInterval(() => {
        timerSecondsLeft--;
        secEl.textContent = timerSecondsLeft;
        if (timerSecondsLeft <= 3) display.classList.add('urgent');
        if (timerSecondsLeft <= 0) { clearQuizTimer(); handleTimeout(correct); }
    }, 1000);
}

function clearQuizTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    const d = document.getElementById('timer-display');
    if (d) { d.style.display = 'none'; d.classList.remove('urgent'); }
}

function updateScoreDisplay() {
    document.getElementById('score-correct').textContent = score;
    document.getElementById('score-wrong').textContent = wrongAnswers.length;
    document.getElementById('score-progress').textContent = currentQuestionIndex + 1;
}

// ============================================================================
// 7. RESULTS
// ============================================================================
function showResults({ correctCount, total, wrongList, mode }) {
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('score-display').style.display = 'none';
    document.getElementById('quiz-progress-bar').style.display = 'none';

    historyStack.push('screen-results');
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById('screen-results').classList.add('active');
    document.getElementById('header-title').innerText = "Kết Quả";

    const percent = total > 0 ? Math.round((correctCount / total) * 100) : 0;
    const circumference = 2 * Math.PI * 64;
    const offset = circumference - (percent / 100) * circumference;

    let color, msg, sub;
    if (percent >= 80) { color = 'var(--correct)'; msg = "Xuất sắc!"; sub = mode === 'challenge' ? "Bạn thực hiện ký hiệu chuẩn xác." : "Bạn nắm vững ngôn ngữ ký hiệu rất tốt."; }
    else if (percent >= 50) { color = 'var(--warning)'; msg = "Khá tốt!"; sub = "Hãy luyện tập thêm những từ còn sai."; }
    else { color = 'var(--wrong)'; msg = "Cần cố gắng hơn!"; sub = "Xem lại các từ sai bên dưới và thử lại nhé."; }

    let wrongHTML = '';
    if (wrongList.length > 0) {
        wrongHTML = `
            <div class="wrong-review">
                <div class="wrong-review-title">
                    <i class="fas fa-exclamation-circle"></i> Các từ cần ôn lại (${wrongList.length})
                </div>
                ${wrongList.map(w => `
                    <div class="wrong-item">
                        <span><strong>${w.question}</strong></span>
                        <span>
                            <span class="your-ans">${w.yourAnswer}</span>
                            → <span class="correct-ans">${w.correctAnswer}</span>
                        </span>
                    </div>
                `).join('')}
            </div>`;
    }

    document.getElementById('results-box').innerHTML = `
        <div class="results-score-ring">
            <svg viewBox="0 0 144 144">
                <circle class="track" cx="72" cy="72" r="64"/>
                <circle class="fill-ring" cx="72" cy="72" r="64"
                    stroke="${color}"
                    stroke-dasharray="${circumference}"
                    stroke-dashoffset="${circumference}"
                    id="score-ring-fill"/>
            </svg>
            <div class="results-score-text">
                <span class="number" style="color: ${color}">${correctCount}/${total}</span>
                <span class="label">${percent}%</span>
            </div>
        </div>
        <div class="results-message" style="color: ${color}">${msg}</div>
        <div class="results-subtitle">${sub}</div>
        ${wrongHTML}
        <div class="results-actions">
            <button class="results-btn secondary" onclick="goHome()">
                <i class="fas fa-home"></i> Trang chủ
            </button>
            <button class="results-btn primary" onclick="${mode === 'challenge' ? 'retryChallenge()' : 'retryQuiz()'}">
                <i class="fas fa-redo"></i> Chơi lại
            </button>
        </div>`;

    requestAnimationFrame(() => requestAnimationFrame(() => {
        const ring = document.getElementById('score-ring-fill');
        if (ring) ring.style.strokeDashoffset = offset;
    }));
}

function showQuizResults() {
    showResults({ correctCount: score, total: currentQuizPool.length, wrongList: wrongAnswers, mode: 'quiz' });
}

function showChallengeResults() {
    showResults({ correctCount: challengeScore, total: challengePool.length, wrongList: challengeWrong, mode: 'challenge' });
}

window.goHome = function () {
    historyStack = ['screen-main-menu'];
    switchScreen('screen-main-menu', true);
};
window.retryQuiz = function () {
    historyStack = ['screen-main-menu', 'screen-quiz-setup'];
    switchScreen('screen-quiz-setup', true);
};
window.retryChallenge = function () {
    historyStack = ['screen-main-menu', 'screen-challenge-setup'];
    switchScreen('screen-challenge-setup', true);
};

// ============================================================================
// 8. PRACTICE ENTRY
// ============================================================================
window.startPractice = function (mode) {
    if (mode === 'challenge') { switchScreen('screen-challenge-setup'); return; }

    practiceMode = mode;
    const targetWordObj = document.getElementById('target-word');
    const modal = document.getElementById('custom-prompt-modal');
    const input = document.getElementById('custom-prompt-input');

    if (mode === 'specific') {
        modal.classList.add('active');
        input.value = '';
        const quickPick = document.getElementById('vocab-quick-pick');
        quickPick.innerHTML = allVocabs.map(v =>
            `<button class="vocab-pick-btn" data-word="${v}">${v}</button>`
        ).join('');

        quickPick.onclick = function (e) {
            const btn = e.target.closest('.vocab-pick-btn');
            if (!btn) return;
            modal.classList.remove('active');
            quickPick.onclick = null;
            targetWordObj.innerText = btn.dataset.word;
            switchScreen('screen-practice');
        };
        setTimeout(() => input.focus(), 300);

        document.getElementById('modal-submit-btn').onclick = () => {
            const word = input.value.trim().toUpperCase();
            if (!word) {
                input.style.borderColor = 'var(--wrong)';
                setTimeout(() => input.style.borderColor = '#E2E8F0', 1000);
                return;
            }
            if (!allVocabs.includes(word)) {
                input.style.borderColor = 'var(--wrong)';
                input.placeholder = 'Từ không hợp lệ!';
                input.value = '';
                setTimeout(() => {
                    input.style.borderColor = '#E2E8F0';
                    input.placeholder = 'VD: CHÀO, ĂN, BẠN...';
                }, 2000);
                return;
            }
            modal.classList.remove('active');
            targetWordObj.innerText = word;
            switchScreen('screen-practice');
        };
        input.onkeyup = (e) => { if (e.key === 'Enter') document.getElementById('modal-submit-btn').click(); };
        document.getElementById('modal-cancel-btn').onclick = () => modal.classList.remove('active');

    } else if (mode === 'random-single') {
        targetWordObj.innerText = allVocabs[Math.floor(Math.random() * allVocabs.length)];
        switchScreen('screen-practice');
    }
};

// ============================================================================
// 9. PRACTICE SESSION (THIN CLIENT)
// ============================================================================
async function initPracticeSession() {
    const video = document.getElementById('webcam-video');
    const statusEl = document.getElementById('camera-status');

    practiceVideoEl = video;

    // Offscreen canvas for JPEG capture (one allocation, reused every frame)
    if (!captureCanvas) {
        captureCanvas = document.createElement('canvas');
        captureCanvas.width = CAMERA_WIDTH;
        captureCanvas.height = CAMERA_HEIGHT;
        captureCtx = captureCanvas.getContext('2d');
    }

    // Reset UI
    updateStatePill('connecting', 'Đang kết nối Camera...', 'fa-spinner fa-spin');
    document.getElementById('collect-bar').style.width = '0%';
    document.getElementById('countdown-overlay').textContent = '';
    document.getElementById('countdown-overlay').classList.remove('active');
    document.getElementById('success-overlay').classList.remove('active');
    updatePredCards([]);
    document.getElementById('btn-retry').style.display = 'none';
    document.getElementById('btn-new-word').style.display = 'none';
    const skipBtn = document.getElementById('btn-skip-challenge');
    if (skipBtn) skipBtn.style.display = practiceMode === 'challenge' ? 'inline-flex' : 'none';

    setupChallengeReference();

    // Open camera (lower resolution = faster server inference)
    statusEl.style.display = 'flex';
    statusEl.className = 'camera-status connecting';
    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang kết nối Camera...';

    try {
        practiceStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CAMERA_WIDTH },
                height: { ideal: CAMERA_HEIGHT },
                facingMode: 'user'
            }
        });
        video.srcObject = practiceStream;
        await video.play();
    } catch (err) {
        statusEl.innerHTML = `<i class="fas fa-times"></i> Không mở được camera: ${err.message}`;
        statusEl.className = 'camera-status error';
        updateStatePill('error', 'Lỗi Camera', 'fa-exclamation-triangle');
        return;
    }
    statusEl.style.display = 'none';

    // Open WebSocket
    try {
        practiceWS = new WebSocket(WS_URL);
    } catch (err) {
        updateStatePill('error', 'Lỗi WebSocket', 'fa-exclamation-triangle');
        return;
    }

    practiceWS.onopen = () => {
        updateStatePill('idle', 'Đã kết nối', 'fa-check-circle');
        const targetWord = document.getElementById('target-word').innerText.trim().toUpperCase();
        practiceWS.send(JSON.stringify({ type: 'start', target_word: targetWord }));
        inflightFrameTs = 0;
        startCaptureLoop();
    };
    practiceWS.onmessage = (evt) => {
        try { handleWSMessage(JSON.parse(evt.data)); } catch (e) { console.error('WS parse:', e); }
    };
    practiceWS.onclose = () => updateStatePill('idle', 'Đã ngắt kết nối', 'fa-plug');
    practiceWS.onerror = () => updateStatePill('error', 'Lỗi kết nối Server', 'fa-exclamation-triangle');
}

// ============================================================================
// 10. CAPTURE LOOP — Send-Then-Wait pattern (key to no-lag)
// ============================================================================
function startCaptureLoop() {
    if (practiceRAF) cancelAnimationFrame(practiceRAF);

    function tick() {
        practiceRAF = requestAnimationFrame(tick);

        const video = practiceVideoEl;
        if (!video || video.readyState < 2) return;
        if (!practiceWS || practiceWS.readyState !== WebSocket.OPEN) return;

        const now = performance.now();

        // RULE 1: Drop stale inflight frame (server timed out)
        if (inflightFrameTs > 0 && now - inflightFrameTs > FRAME_STALENESS_MS) {
            inflightFrameTs = 0;
        }

        // RULE 2: One frame in network at a time. If still waiting, skip.
        if (inflightFrameTs > 0) return;

        // RULE 3: Min interval cap to avoid spamming when server is fast
        if (now - lastSendTime < FRAME_MAX_INTERVAL) return;

        // Capture latest video frame at low resolution
        try {
            captureCtx.drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
            const dataUrl = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
            practiceWS.send(JSON.stringify({ type: 'frame', image: dataUrl }));
            inflightFrameTs = now;
            lastSendTime = now;
        } catch (e) {
            console.error('Capture error:', e);
        }
    }
    practiceRAF = requestAnimationFrame(tick);
}

// ============================================================================
// 11. WS MESSAGE HANDLER + SKELETON DRAWING
// ============================================================================
function handleWSMessage(data) {
    // Any response (status/result) clears the inflight slot
    if (data.type === 'status' || data.type === 'result') {
        inflightFrameTs = 0;
    }

    if (data.type === 'started') {
        updateStatePill('wait', 'Chuẩn bị...', 'fa-hourglass-half');
        return;
    }

    if (data.type === 'result' && data.success) {
        document.getElementById('success-overlay').classList.add('active');
        updateStatePill('success', 'CHÍNH XÁC!', 'fa-check-circle');
        if (data.top3) updatePredCards(data.top3);
        onPracticeSuccess();
        return;
    }

    if (data.type === 'stopped') {
        updateStatePill('idle', 'Đã dừng', 'fa-stop-circle');
        return;
    }
    if (data.type === 'error') {
        updateStatePill('error', data.message || 'Lỗi', 'fa-exclamation-triangle');
        return;
    }
    if (data.type !== 'status') return;

    const state = data.state || '';
    switch (state) {
        case 'WAIT': {
            updateStatePill('wait', data.message || 'Chuẩn bị...', 'fa-hourglass-half');
            const cd = data.countdown || 0;
            const overlay = document.getElementById('countdown-overlay');
            if (cd > 0.5) { overlay.textContent = Math.ceil(cd); overlay.classList.add('active'); }
            else overlay.classList.remove('active');
            document.getElementById('collect-bar').style.width = '0%';
            break;
        }
        case 'COLLECT':
            updateStatePill('collect', data.message || 'Thu thập...', 'fa-hand-sparkles');
            document.getElementById('countdown-overlay').classList.remove('active');
            document.getElementById('collect-bar').style.width = `${(data.progress || 0) * 100}%`;
            break;
        case 'PREDICT':
            updateStatePill('predict', 'Đang phân tích...', 'fa-brain');
            document.getElementById('collect-bar').style.width = '100%';
            break;
        case 'SHOW':
            updateStatePill('show', data.message || 'Xem kết quả...', 'fa-eye');
            break;
    }
    if (data.top3 && data.top3.length > 0) updatePredCards(data.top3);
}

// ============================================================================
// 12. CHALLENGE FLOW
// ============================================================================
function setupChallengeReference() {
    const refContainer = document.getElementById('challenge-reference');
    const stage = document.querySelector('.practice-stage');
    if (!refContainer) return;
    if (practiceMode === 'challenge') {
        const word = challengePool[challengeIndex];
        if (word) {
            refContainer.style.display = 'flex';
            stage?.classList.add('has-reference');
            refContainer.innerHTML = `
                <div class="ref-header">
                    <i class="fas fa-video"></i> Ký hiệu mẫu <strong>${word.correctLabel}</strong>
                </div>
                <video autoplay loop muted playsinline class="ref-video">
                    <source src="${word.videoUrl}" type="video/mp4">
                </video>`;
        }
    } else {
        refContainer.style.display = 'none';
        stage?.classList.remove('has-reference');
        refContainer.innerHTML = '';
    }
}

function onPracticeSuccess() {
    if (practiceMode !== 'challenge') {
        document.getElementById('btn-retry').style.display = 'inline-flex';
        document.getElementById('btn-new-word').style.display = 'inline-flex';
        return;
    }
    challengeScore++;
    challengeIndex++;
    setTimeout(() => advanceChallenge(true), 1500);
}

function advanceChallenge(success) {
    if (!success) {
        const word = challengePool[challengeIndex];
        if (word) {
            challengeWrong.push({
                question: word.correctLabel,
                yourAnswer: '(Bỏ qua)',
                correctAnswer: word.correctLabel
            });
            challengeIndex++;
        }
    }

    if (challengeIndex >= challengePool.length) {
        stopPracticeSession();
        showChallengeResults();
        return;
    }

    const nextWord = challengePool[challengeIndex];
    document.getElementById('target-word').innerText = nextWord.correctLabel;
    document.getElementById('header-title').innerText = `Thử Thách ${challengeIndex + 1}/${challengePool.length}`;
    document.getElementById('success-overlay').classList.remove('active');
    setupChallengeReference();

    if (practiceWS && practiceWS.readyState === WebSocket.OPEN) {
        practiceWS.send(JSON.stringify({ type: 'start', target_word: nextWord.correctLabel }));
    }
}

window.skipChallengeWord = function () {
    if (practiceMode === 'challenge') advanceChallenge(false);
};

// ============================================================================
// 13. UI HELPERS
// ============================================================================
function updateStatePill(stateClass, text, iconClass) {
    const pill = document.getElementById('state-pill');
    if (!pill) return;
    pill.className = `practice-state-pill ${stateClass}`;
    pill.innerHTML = `<i class="fas ${iconClass}"></i> ${text}`;
}

function updatePredCards(top3) {
    for (let i = 0; i < 3; i++) {
        const card = document.getElementById(`pred-${i + 1}`);
        if (!card) continue;
        if (i < top3.length) {
            card.querySelector('.word').textContent = top3[i].labelVn || top3[i].label;
            card.querySelector('.score').textContent = top3[i].score + '%';
            card.classList.toggle('highlight', i === 0 && top3[i].score >= 40);
        } else {
            card.querySelector('.word').textContent = '—';
            card.querySelector('.score').textContent = '0%';
            card.classList.remove('highlight');
        }
    }
}

// ============================================================================
// 14. PRACTICE CONTROLS
// ============================================================================
window.stopPractice = function () {
    stopPracticeSession();
    if (historyStack.length > 1) {
        historyStack.pop();
        switchScreen(historyStack[historyStack.length - 1], true);
    }
};

function stopPracticeSession() {
    if (practiceRAF) { cancelAnimationFrame(practiceRAF); practiceRAF = null; }
    inflightFrameTs = 0;
    if (practiceWS) {
        try {
            if (practiceWS.readyState === WebSocket.OPEN) {
                practiceWS.send(JSON.stringify({ type: 'stop' }));
            }
            practiceWS.close();
        } catch (e) {}
        practiceWS = null;
    }
    if (practiceStream) {
        practiceStream.getTracks().forEach(t => t.stop());
        practiceStream = null;
    }
    const video = document.getElementById('webcam-video');
    if (video) video.srcObject = null;
    const cdOverlay = document.getElementById('countdown-overlay');
    if (cdOverlay) cdOverlay.classList.remove('active');
    const successOverlay = document.getElementById('success-overlay');
    if (successOverlay) successOverlay.classList.remove('active');
}

window.retryPractice = function () {
    document.getElementById('success-overlay').classList.remove('active');
    document.getElementById('btn-retry').style.display = 'none';
    document.getElementById('btn-new-word').style.display = 'none';
    if (practiceWS && practiceWS.readyState === WebSocket.OPEN) {
        const t = document.getElementById('target-word').innerText.trim().toUpperCase();
        practiceWS.send(JSON.stringify({ type: 'start', target_word: t }));
    } else {
        initPracticeSession();
    }
};

window.goBackFromPractice = function () {
    stopPracticeSession();
    historyStack = ['screen-main-menu', 'screen-practice-menu'];
    switchScreen('screen-practice-menu', true);
};

// ============================================================================
// 15. UTILITIES
// ============================================================================
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// ============================================================================
// 16. INIT
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initChips();
});
