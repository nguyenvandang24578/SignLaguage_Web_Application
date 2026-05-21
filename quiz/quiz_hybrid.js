// ============================================================================
// quiz_hybrid.js — Sign Language Quiz (HYBRID CLIENT)
// ============================================================================
// IMPORTANT: Dynamic import for MediaPipe — top-level static import would
// HALT module parsing if CDN unreachable, leaving window.switchScreen etc.
// unregistered → all UI clicks would silently fail.
// ============================================================================

// Global error trap — make any error visible (instead of silent module failure)
window.addEventListener('error', (e) => {
    console.error('[GLOBAL ERROR]', e.error || e.message, e);
});
window.addEventListener('unhandledrejection', (e) => {
    console.error('[UNHANDLED PROMISE]', e.reason);
});

console.log('[quiz_hybrid.js] Module loading...');

const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// Lazy-loaded — populated by initMediaPipe()
let FilesetResolver = null;
let PoseLandmarker = null;
let HandLandmarker = null;

async function loadMediaPipeLib() {
    if (FilesetResolver !== null) return;   // already loaded
    console.log('[MediaPipe] Loading library from CDN...');
    try {
        const mod = await import(MEDIAPIPE_CDN);
        FilesetResolver = mod.FilesetResolver;
        PoseLandmarker = mod.PoseLandmarker;
        HandLandmarker = mod.HandLandmarker;
        console.log('[MediaPipe] Library loaded.');
    } catch (err) {
        console.error('[MediaPipe] Failed to load library from CDN:', err);
        throw err;
    }
}

// ============================================================================
// 1. DATA + TOPIC CLASSIFICATION (giữ nguyên từ quiz.js cũ)
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
    { id: 16, correctLabel: "ĐÔI DÉP",       videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/doidep.mp4",        topic: "object" },
    { id: 17, correctLabel: "ĐỨNG",          videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/dung.mp4",          topic: "action" },
    { id: 18, correctLabel: "GĂNG TAY",      videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/gangtay.mp4",       topic: "object" },
    { id: 19, correctLabel: "GẦY",           videoUrl: "https://huggingface.co/datasets/DangNguyenVan258/signvn-data/resolve/main/learn/videos/gay.mp4",           topic: "body" },
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
// 2. SERVER CONFIG
// ============================================================================
const SERVER_HOST = window.location.hostname || '127.0.0.1';
const SERVER_PORT = 8001;                          // server_hybrid.py default port
const SERVER_BASE = `http://${SERVER_HOST}:${SERVER_PORT}`;
const WS_URL = `ws://${SERVER_HOST}:${SERVER_PORT}/ws/practice`;


// ============================================================================
// 3. TRAINING CONVENTION CONSTANTS (must match server_hybrid.py /model_info)
// ============================================================================
const NUM_JOINTS = 27;

// Pose: nose, L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist
const MP_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16];

// Hand: wrist, thumb_tip, index_mcp/tip, middle_mcp/tip, ring_mcp/tip, pinky_mcp/tip
const MP_HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];

// CRITICAL: training data was extracted at 1280×720 pixel space
// Browser MUST multiply normalized landmarks by these constants before sending,
// regardless of actual camera resolution or canvas size.
const MODEL_INPUT_WIDTH = 1280;
const MODEL_INPUT_HEIGHT = 720;


// ============================================================================
// 4. CAMERA & MEDIAPIPE CONFIG
// ============================================================================
const CAMERA_WIDTH = 1280;       // For display, can be higher (Full HD if camera supports)
const CAMERA_HEIGHT = 720;
const TARGET_FPS = 30;           // MediaPipe inference + WS send target


// ============================================================================
// 5. STATE
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

// MediaPipe & camera
let practiceWS = null;
let practiceStream = null;
let practiceVideoEl = null;
let practiceCanvasEl = null;
let practiceCanvasCtx = null;
let practiceRAF = null;
let poseLandmarker = null;
let handLandmarker = null;
let mediapipeReady = false;
let lastVideoTime = -1;
let practiceState = 'IDLE';      // mirror of server state

// FPS counter
let frameCount = 0;
let fpsLastTime = performance.now();
let currentFPS = 0;


// ============================================================================
// 6. ONE-EURO FILTER (port từ server.py)
// ============================================================================
class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.xPrev = null;     // Float32Array of shape (N*2,) flat
        this.dxPrev = null;
        this.tPrev = 0;
    }

    static smoothingFactor(te, cutoff) {
        const r = 2 * Math.PI * cutoff * te;
        return r / (r + 1);
    }

    /**
     * x: Float32Array of length 2N (interleaved x,y for N points)
     * t: timestamp in ms
     * Returns: new Float32Array filtered in-place style
     */
    filter(x, t) {
        if (this.xPrev === null) {
            this.xPrev = new Float32Array(x);
            this.dxPrev = new Float32Array(x.length);
            this.tPrev = t;
            return new Float32Array(x);
        }

        const te = Math.max((t - this.tPrev) / 1000, 1e-6);   // ms → sec

        const dx = new Float32Array(x.length);
        const dxHat = new Float32Array(x.length);
        const xHat = new Float32Array(x.length);

        const aD = OneEuroFilter.smoothingFactor(te, this.dCutoff);

        for (let i = 0; i < x.length; i++) {
            dx[i] = (x[i] - this.xPrev[i]) / te;
            dxHat[i] = aD * dx[i] + (1 - aD) * this.dxPrev[i];

            const cutoff = this.minCutoff + this.beta * Math.abs(dxHat[i]);
            const a = OneEuroFilter.smoothingFactor(te, cutoff);
            xHat[i] = a * x[i] + (1 - a) * this.xPrev[i];
        }

        this.xPrev = xHat;
        this.dxPrev = dxHat;
        this.tPrev = t;
        return xHat;
    }

    reset() {
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = 0;
    }
}

// 3 filters tách biệt cho pose / left hand / right hand
const poseFilter   = new OneEuroFilter(1.0, 0.02, 1.0);   // pose ít jitter, smooth nhẹ
const lHandFilter  = new OneEuroFilter(1.5, 0.05, 1.0);   // hand nhiều jitter + cần bám sát
const rHandFilter  = new OneEuroFilter(1.5, 0.05, 1.0);


// ============================================================================
// 7. MEDIAPIPE INITIALIZATION
// ============================================================================
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

async function initMediaPipe() {
    if (mediapipeReady) return;
    await loadMediaPipeLib();   // ensure library is loaded first

    console.log('[MediaPipe] Initializing models...');
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    [poseLandmarker, handLandmarker] = await Promise.all([
        PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
        }),
        HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
        }),
    ]);
    mediapipeReady = true;
    console.log('[MediaPipe] Ready (GPU delegate).');
}


// ============================================================================
// 8. KEYPOINT EXTRACTION — MUST match training convention exactly
// ============================================================================
/**
 * Extract 27 keypoints theo convention training (sign_27.py).
 *
 * @param {Object} poseResult - PoseLandmarker.detectForVideo() result
 * @param {Object} handResult - HandLandmarker.detectForVideo() result
 * @returns {Float32Array} flat array of length 27*3 = 81 (x, y, vis × 27 joints)
 */
function extract27Keypoints(poseResult, handResult) {
    const pts = new Float32Array(NUM_JOINTS * 3);  // 27 × (x, y, vis), default 0

    // ─── POSE (slots 0-6) ───
    if (poseResult.landmarks && poseResult.landmarks.length > 0) {
        const poseLandmarks = poseResult.landmarks[0];
        for (let i = 0; i < MP_POSE_INDICES.length; i++) {
            const mpIdx = MP_POSE_INDICES[i];
            const lm = poseLandmarks[mpIdx];
            if (lm) {
                pts[i * 3 + 0] = lm.x * MODEL_INPUT_WIDTH;
                pts[i * 3 + 1] = lm.y * MODEL_INPUT_HEIGHT;
                pts[i * 3 + 2] = lm.visibility ?? 0.0;
            }
        }
    }

    // ─── HANDS (slots 7-16 = LEFT, slots 17-26 = RIGHT) ───
    // HandLandmarker trả nhiều hands trong 1 array; phải dùng handedness để biết
    // hand nào là Left / Right.
    if (handResult.landmarks && handResult.handednesses) {
        for (let h = 0; h < handResult.landmarks.length; h++) {
            const handLandmarks = handResult.landmarks[h];
            const handednessCategory = handResult.handednesses[h][0]; // top-1 category
            const isLeft = handednessCategory.categoryName === "Left";
            const slotStart = isLeft ? 7 : 17;

            for (let i = 0; i < MP_HAND_INDICES.length; i++) {
                const mpIdx = MP_HAND_INDICES[i];
                const lm = handLandmarks[mpIdx];
                if (lm) {
                    pts[(slotStart + i) * 3 + 0] = lm.x * MODEL_INPUT_WIDTH;
                    pts[(slotStart + i) * 3 + 1] = lm.y * MODEL_INPUT_HEIGHT;
                    pts[(slotStart + i) * 3 + 2] = 1.0;   // training hardcoded 1.0
                }
            }
        }
    }

    return pts;
}


/**
 * Apply One-Euro Filter to keypoints in pixel space.
 * pts: Float32Array length 81 (x,y,vis × 27)
 * Returns: smoothed Float32Array of same length.
 */
function smoothKeypoints(pts, tNow) {
    // Tách x,y vào 3 buffer flat tương ứng pose/lhand/rhand để filter tách biệt.
    // Slot 0-6 = pose (7 joints), 7-16 = lhand (10), 17-26 = rhand (10).

    const poseFlat = new Float32Array(7 * 2);
    const lhandFlat = new Float32Array(10 * 2);
    const rhandFlat = new Float32Array(10 * 2);

    let poseHasData = false, lhandHasData = false, rhandHasData = false;

    for (let i = 0; i < 7; i++) {
        poseFlat[i * 2 + 0] = pts[i * 3 + 0];
        poseFlat[i * 2 + 1] = pts[i * 3 + 1];
        if (pts[i * 3 + 0] !== 0 || pts[i * 3 + 1] !== 0) poseHasData = true;
    }
    for (let i = 0; i < 10; i++) {
        lhandFlat[i * 2 + 0] = pts[(7 + i) * 3 + 0];
        lhandFlat[i * 2 + 1] = pts[(7 + i) * 3 + 1];
        if (pts[(7 + i) * 3 + 0] !== 0 || pts[(7 + i) * 3 + 1] !== 0) lhandHasData = true;

        rhandFlat[i * 2 + 0] = pts[(17 + i) * 3 + 0];
        rhandFlat[i * 2 + 1] = pts[(17 + i) * 3 + 1];
        if (pts[(17 + i) * 3 + 0] !== 0 || pts[(17 + i) * 3 + 1] !== 0) rhandHasData = true;
    }

    // Filter only if hand detected (reset filter when hand disappears for clean restart)
    if (poseHasData) {
        const sPose = poseFilter.filter(poseFlat, tNow);
        for (let i = 0; i < 7; i++) {
            pts[i * 3 + 0] = sPose[i * 2 + 0];
            pts[i * 3 + 1] = sPose[i * 2 + 1];
        }
    } else {
        poseFilter.reset();
    }
    if (lhandHasData) {
        const sL = lHandFilter.filter(lhandFlat, tNow);
        for (let i = 0; i < 10; i++) {
            pts[(7 + i) * 3 + 0] = sL[i * 2 + 0];
            pts[(7 + i) * 3 + 1] = sL[i * 2 + 1];
        }
    } else {
        lHandFilter.reset();
    }
    if (rhandHasData) {
        const sR = rHandFilter.filter(rhandFlat, tNow);
        for (let i = 0; i < 10; i++) {
            pts[(17 + i) * 3 + 0] = sR[i * 2 + 0];
            pts[(17 + i) * 3 + 1] = sR[i * 2 + 1];
        }
    } else {
        rHandFilter.reset();
    }
    return pts;
}


// ============================================================================
// 9. SKELETON DRAWING (canvas overlay)
// ============================================================================
// Connections defined in pixel space (after × MODEL_INPUT_WIDTH/HEIGHT),
// rendered scaled to canvas size.
//
// Graph edges từ sign_27.py:
const POSE_CONNECTIONS = [
    [0, 1], [0, 2],          // nose → shoulders
    [1, 3], [3, 5],          // R shoulder → elbow → wrist (slot numbers per training)
    [2, 4], [4, 6],          // L shoulder → elbow → wrist
];

const HAND_CONNECTIONS_LOCAL = [
    [0, 1],                  // wrist → thumb_tip
    [0, 2], [2, 3],          // wrist → index_mcp → index_tip
    [0, 4], [4, 5],          // wrist → middle_mcp → middle_tip
    [0, 6], [6, 7],          // wrist → ring_mcp → ring_tip
    [0, 8], [8, 9],          // wrist → pinky_mcp → pinky_tip
];

function drawSkeleton(pts) {
    const canvas = practiceCanvasEl;
    const ctx = practiceCanvasCtx;
    if (!canvas || !ctx) return;

    // Match canvas to video element display size
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    // Camera CSS mirror via transform: scaleX(-1) — but pts are in unmirrored space.
    // We mirror the drawing too so skeleton aligns with mirrored video.
    ctx.save();
    ctx.translate(cw, 0);
    ctx.scale(-1, 1);

    const sx = cw / MODEL_INPUT_WIDTH;
    const sy = ch / MODEL_INPUT_HEIGHT;

    // Helper to read point
    const getPoint = (slot) => ({
        x: pts[slot * 3 + 0] * sx,
        y: pts[slot * 3 + 1] * sy,
        ok: !(pts[slot * 3 + 0] === 0 && pts[slot * 3 + 1] === 0),
    });

    // Pose lines
    ctx.strokeStyle = '#00C896';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    for (const [a, b] of POSE_CONNECTIONS) {
        const pa = getPoint(a), pb = getPoint(b);
        if (!pa.ok || !pb.ok) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
    }

    // Pose joints
    ctx.fillStyle = '#00C896';
    for (let i = 0; i < 7; i++) {
        const p = getPoint(i);
        if (!p.ok) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    // Hands — left (slots 7-16) and right (slots 17-26)
    const drawHand = (offset, color) => {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2.5;
        for (const [a, b] of HAND_CONNECTIONS_LOCAL) {
            const pa = getPoint(offset + a), pb = getPoint(offset + b);
            if (!pa.ok || !pb.ok) continue;
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
        }
        for (let i = 0; i < 10; i++) {
            const p = getPoint(offset + i);
            if (!p.ok) continue;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }
    };
    drawHand(7,  '#22C55E');   // L hand → green
    drawHand(17, '#FB923C');   // R hand → orange

    ctx.restore();

    // FPS counter (NOT mirrored — overlay UI text)
    ctx.fillStyle = '#10B981';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${currentFPS} FPS`, cw - 12, 24);
}


// ============================================================================
// 10. MAIN LOOP — capture video → MediaPipe → extract → smooth → draw → send
// ============================================================================
async function captureLoop() {
    if (!practiceVideoEl || practiceVideoEl.readyState < 2) {
        practiceRAF = requestAnimationFrame(captureLoop);
        return;
    }
    if (!mediapipeReady) {
        practiceRAF = requestAnimationFrame(captureLoop);
        return;
    }

    const nowMs = performance.now();
    const videoTime = practiceVideoEl.currentTime * 1000;

    // Skip duplicate frame (video hasn't advanced)
    if (videoTime !== lastVideoTime) {
        lastVideoTime = videoTime;

        // Run both models on same frame timestamp
        const poseResult = poseLandmarker.detectForVideo(practiceVideoEl, nowMs);
        const handResult = handLandmarker.detectForVideo(practiceVideoEl, nowMs);

        // Extract 27 kpts in training pixel space
        let pts = extract27Keypoints(poseResult, handResult);

        // Apply One-Euro smoothing
        pts = smoothKeypoints(pts, nowMs);

        // Draw on overlay canvas
        drawSkeleton(pts);

        // Send to server ONLY when in COLLECT state (server signals via WS)
        if (practiceWS && practiceWS.readyState === WebSocket.OPEN
            && practiceState === 'COLLECT') {
            // Convert to Array for JSON serialization
            // Shape: [[x,y,vis], [x,y,vis], ...] × 27
            const joints = [];
            for (let i = 0; i < NUM_JOINTS; i++) {
                joints.push([
                    pts[i * 3 + 0],
                    pts[i * 3 + 1],
                    pts[i * 3 + 2],
                ]);
            }
            practiceWS.send(JSON.stringify({ type: 'keypoints', joints }));
        }

        // FPS tracking
        frameCount++;
        if (nowMs - fpsLastTime >= 1000) {
            currentFPS = Math.round(frameCount * 1000 / (nowMs - fpsLastTime));
            frameCount = 0;
            fpsLastTime = nowMs;
        }
    }

    practiceRAF = requestAnimationFrame(captureLoop);
}


// ============================================================================
// 11. CHIP SELECTION (giữ nguyên)
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
// 12. SCREEN NAVIGATION (giữ nguyên từ quiz.js cũ)
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
                : "AI Tracking (Hybrid)";
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
// 13. QUIZ ENGINE (unchanged)
// ============================================================================
function startNewQuizSession() {
    currentQuestionIndex = 0; score = 0; wrongAnswers = [];
    updateScoreDisplay();
    let pool = validQuizData;
    if (quizSettings.topic !== 'all') pool = validQuizData.filter(q => q.topic === quizSettings.topic);
    if (pool.length < 4) {
        document.getElementById('video-container').innerHTML =
            '<div class="video-placeholder"><i class="fas fa-exclamation-triangle"></i><p>Chủ đề này cần ít nhất 4 từ.</p></div>';
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
        b.style.pointerEvents = 'none'; b.classList.add('answered');
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
// 14. RESULTS (unchanged)
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
        wrongHTML = `<div class="wrong-review">
            <div class="wrong-review-title"><i class="fas fa-exclamation-circle"></i> Các từ cần ôn lại (${wrongList.length})</div>
            ${wrongList.map(w => `<div class="wrong-item">
                <span><strong>${w.question}</strong></span>
                <span><span class="your-ans">${w.yourAnswer}</span> → <span class="correct-ans">${w.correctAnswer}</span></span>
            </div>`).join('')}
        </div>`;
    }

    document.getElementById('results-box').innerHTML = `
        <div class="results-score-ring">
            <svg viewBox="0 0 144 144">
                <circle class="track" cx="72" cy="72" r="64"/>
                <circle class="fill-ring" cx="72" cy="72" r="64" stroke="${color}"
                    stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"
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
            <button class="results-btn secondary" onclick="goHome()"><i class="fas fa-home"></i> Trang chủ</button>
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
// 15. PRACTICE ENTRY (unchanged)
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
// 16. PRACTICE SESSION — HYBRID INIT
// ============================================================================
async function initPracticeSession() {
    const wrapper = document.getElementById('camera-wrapper');
    const statusEl = document.getElementById('camera-status');
    practiceVideoEl = document.getElementById('webcam-video');
    practiceCanvasEl = document.getElementById('skeleton-canvas');

    // Reset UI
    updateStatePill('connecting', 'Khởi tạo MediaPipe...', 'fa-spinner fa-spin');
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

    // ─── 1. Init MediaPipe (load WASM + 2 models) ───
    try {
        await initMediaPipe();
    } catch (err) {
        console.error('MediaPipe init failed:', err);
        updateStatePill('error', 'Lỗi MediaPipe: ' + err.message, 'fa-exclamation-triangle');
        return;
    }

    // ─── 2. Open camera ───
    statusEl.style.display = 'flex';
    statusEl.className = 'camera-status connecting';
    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang mở Camera...';

    try {
        practiceStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CAMERA_WIDTH },
                height: { ideal: CAMERA_HEIGHT },
                frameRate: { ideal: TARGET_FPS },
                facingMode: 'user'
            }
        });
        practiceVideoEl.srcObject = practiceStream;
        await practiceVideoEl.play();

        // Wait for video to have valid dimensions
        await new Promise((res) => {
            if (practiceVideoEl.videoWidth > 0) return res();
            practiceVideoEl.onloadedmetadata = () => res();
        });

        // Match canvas to video display size
        const rect = wrapper.getBoundingClientRect();
        practiceCanvasEl.width = rect.width;
        practiceCanvasEl.height = rect.height;
        practiceCanvasCtx = practiceCanvasEl.getContext('2d');
    } catch (err) {
        statusEl.innerHTML = `<i class="fas fa-times"></i> Không mở được camera: ${err.message}`;
        statusEl.className = 'camera-status error';
        updateStatePill('error', 'Lỗi Camera', 'fa-exclamation-triangle');
        return;
    }
    statusEl.style.display = 'none';

    // Reset filters
    poseFilter.reset();
    lHandFilter.reset();
    rHandFilter.reset();

    // ─── 3. Open WebSocket to hybrid server ───
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
    };
    practiceWS.onmessage = (evt) => {
        try { handleWSMessage(JSON.parse(evt.data)); } catch (e) { console.error('WS parse:', e); }
    };
    practiceWS.onclose = () => updateStatePill('idle', 'Đã ngắt kết nối', 'fa-plug');
    practiceWS.onerror = () => updateStatePill('error', 'Lỗi kết nối Server', 'fa-exclamation-triangle');

    // ─── 4. Start capture loop ───
    practiceRAF = requestAnimationFrame(captureLoop);
}


// ============================================================================
// 17. WS MESSAGE HANDLER (sync practiceState với server state)
// ============================================================================
function handleWSMessage(data) {
    if (data.type === 'started') {
        practiceState = 'WAIT';
        updateStatePill('wait', 'Chuẩn bị...', 'fa-hourglass-half');
        return;
    }

    if (data.type === 'result') {
        practiceState = data.success ? 'IDLE' : 'SHOW';
        if (data.top3) updatePredCards(data.top3);
        showResultFeedback(data.success);
        if (data.success) {
            onPracticeSuccess();
        }
        if (data.inference_ms !== undefined) {
            console.log(`[Inference] ${data.inference_ms}ms`);
        }
        return;
    }

    if (data.type === 'stopped') {
        practiceState = 'IDLE';
        updateStatePill('idle', 'Đã dừng', 'fa-stop-circle');
        return;
    }
    if (data.type === 'error') {
        updateStatePill('error', data.message || 'Lỗi', 'fa-exclamation-triangle');
        return;
    }

    if (data.type !== 'status') return;
    practiceState = data.state || 'IDLE';

    const cameraWrapper = document.getElementById('camera-wrapper');
    switch (data.state) {
        case 'WAIT': {
            updateStatePill('wait', data.message || 'Chuẩn bị...', 'fa-hourglass-half');
            const cd = data.countdown || 0;
            const overlay = document.getElementById('countdown-overlay');
            if (cd > 0.05) {
                overlay.textContent = cd <= 0.5 ? 'BẮT ĐẦU!' : String(Math.ceil(cd));
                overlay.classList.add('active');
            } else {
                overlay.classList.remove('active');
            }
            document.getElementById('collect-bar').style.width = '0%';
            cameraWrapper?.classList.remove('recording');
            // Reset filters when entering new round
            poseFilter.reset(); lHandFilter.reset(); rHandFilter.reset();
            break;
        }
        case 'COLLECT':
            updateStatePill('collect', data.message || 'Thu thập...', 'fa-hand-sparkles');
            document.getElementById('countdown-overlay').classList.remove('active');
            document.getElementById('collect-bar').style.width = `${(data.progress || 0) * 100}%`;
            cameraWrapper?.classList.add('recording');
            break;
        case 'PREDICT':
            updateStatePill('predict', 'Đang phân tích...', 'fa-brain');
            document.getElementById('collect-bar').style.width = '100%';
            cameraWrapper?.classList.remove('recording');
            break;
        case 'SHOW':
            updateStatePill('show', data.message || 'Xem kết quả...', 'fa-eye');
            cameraWrapper?.classList.remove('recording');
            break;
    }
}


// ============================================================================
// 18. CHALLENGE FLOW (unchanged)
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
                <div class="ref-header"><i class="fas fa-video"></i> Ký hiệu mẫu <strong>${word.correctLabel}</strong></div>
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
// 19. UI HELPERS
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
        const wordEl = card.querySelector('.word');
        if (i < top3.length) {
            wordEl.textContent = top3[i].labelVn || top3[i].label;
            card.classList.toggle('highlight', i === 0 && top3[i].score >= 40);
        } else {
            wordEl.textContent = '—';
            card.classList.remove('highlight');
        }
    }
}

function showResultFeedback(success) {
    const card1 = document.getElementById('pred-1');
    const overlay = document.getElementById('success-overlay');
    if (success) {
        AudioFX.success();
        card1?.classList.add('result-correct');
        card1?.classList.remove('result-wrong');
        if (overlay) {
            overlay.classList.add('active');
            overlay.classList.remove('wrong');
            overlay.innerHTML = `<div class="overlay-icon"><i class="fas fa-check-circle"></i></div>
                <div class="overlay-text">CHÍNH XÁC!</div>`;
        }
    } else {
        AudioFX.wrong();
        card1?.classList.add('result-wrong');
        card1?.classList.remove('result-correct');
    }
    setTimeout(() => {
        card1?.classList.remove('result-correct', 'result-wrong');
    }, 2000);
}


// ============================================================================
// 20. PRACTICE CONTROLS
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
    practiceState = 'IDLE';
    if (practiceVideoEl) practiceVideoEl.srcObject = null;
    if (practiceCanvasCtx) practiceCanvasCtx.clearRect(0, 0,
        practiceCanvasEl.width, practiceCanvasEl.height);
    document.getElementById('countdown-overlay')?.classList.remove('active');
    document.getElementById('success-overlay')?.classList.remove('active');
    document.getElementById('camera-wrapper')?.classList.remove('recording');
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
// 21. UTILITIES + AUDIO (unchanged)
// ============================================================================
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

const AudioFX = (() => {
    let ctx = null;
    let enabled = true;
    function ensureCtx() {
        if (ctx) return ctx;
        try {
            const AC = window.AudioContext || window.webkitAudioContext;
            ctx = new AC();
        } catch (e) { enabled = false; }
        return ctx;
    }
    function beep(freq, duration = 0.12, type = 'sine', volume = 0.18) {
        if (!enabled) return;
        const c = ensureCtx(); if (!c) return;
        if (c.state === 'suspended') c.resume();
        const osc = c.createOscillator();
        const gain = c.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, c.currentTime);
        gain.gain.setValueAtTime(0, c.currentTime);
        gain.gain.linearRampToValueAtTime(volume, c.currentTime + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + duration);
        osc.connect(gain).connect(c.destination);
        osc.start();
        osc.stop(c.currentTime + duration);
    }
    return {
        click() { beep(620, 0.05, 'square', 0.10); },
        success() {
            beep(523.25, 0.10, 'sine', 0.18);
            setTimeout(() => beep(659.25, 0.10, 'sine', 0.18), 90);
            setTimeout(() => beep(783.99, 0.18, 'sine', 0.18), 180);
        },
        wrong() {
            beep(220, 0.20, 'triangle', 0.16);
            setTimeout(() => beep(180, 0.20, 'triangle', 0.14), 110);
        },
    };
})();

document.addEventListener('click', (e) => {
    const target = e.target.closest('button, .card, .chip, .feature-card, .nav-tab');
    if (target && !target.disabled) AudioFX.click();
}, true);


// ============================================================================
// 22. INIT
// ============================================================================

// Sanity check — log để debug nếu click không phản hồi
console.log('[quiz_hybrid.js] Module loaded. Functions exposed:', {
    switchScreen:       typeof window.switchScreen,
    startPractice:      typeof window.startPractice,
    launchQuiz:         typeof window.launchQuiz,
    launchChallenge:    typeof window.launchChallenge,
});

document.addEventListener('DOMContentLoaded', () => {
    console.log('[quiz_hybrid.js] DOMContentLoaded fired');
    initChips();
    // Preload MediaPipe in background (saves time when user enters practice screen)
    // Errors here are NON-FATAL — they should not break UI navigation
    initMediaPipe().catch(err => {
        console.warn('[MediaPipe] Preload failed (will retry on practice screen):', err);
    });
});
