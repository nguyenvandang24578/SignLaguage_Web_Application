// gru_inference.js
// ort is loaded via <script> tag in HTML (classic script), available as window.ort
const ort = window.ort;

// Point wasm files to CDN and disable multi-threading (requires crossOriginIsolated)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
ort.env.wasm.numThreads = 1;

const SEQUENCE_LENGTH = 30;
const THRESHOLD       = 0.7;
const ACTIONS         = ["a", "aw", "aa", "b", "c", "d", "dd", "e", "ee", "g", "h", "i", "k", "l", "m", "n", "o", "ow", "oo", "p", "q", "r", "s", "t", "u", "uw", "v", "x", "y", "huyen", "hoi", "nga", "sac", "nang", "nothing"];

let gruSession   = null;
let frameBuffer  = [];

// FIX BUG 2: Debounce reset — chỉ reset buffer sau khi tay vắng mặt đủ lâu
let noHandCount = 0;
const NO_HAND_RESET_FRAMES = 15;

// FIX: skipFrames — sau khi reset (đúng hoặc sai), bỏ qua N frame đầu tiên
// để counter thực sự hiển thị 0 thay vì ngay lập tức thu thập lại.
// 15 frame × 33ms ≈ 500ms đứng yên ở 0 trước khi bắt đầu thu thập chữ mới.
let skipFrames = 0;

// ─── LOAD MODEL ──────────────────────────────────────────────────────────────
export async function loadGRU(modelPath = "./models/gru.onnx") {
    try {
        gruSession = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu', 'wasm'],
        });

        // Log xem thực tế đang dùng backend nào
        console.log("[GRU] ✅ Model loaded, backend:", gruSession.handler?.backend ?? 'unknown');

    } catch (e) {
        console.error("[GRU] ❌ Failed to load model:", e);
    }
}
export async function pushKeypointsAndPredict(landmarks) {
    if (!gruSession) {
        console.warn("[GRU] ⚠️ Session not ready — model chưa load xong!");
        return null;
    }

    // FIX: Nếu đang trong giai đoạn skip, trả về buffering(0) mà không push frame nào.
    // Điều này đảm bảo counter ở màn hình hiển thị 0/30 trong ~500ms sau mỗi lần reset,
    // dù tay có còn trong frame hay không — tránh hiện tượng counter "nhảy từ 10-15".
    if (skipFrames > 0) {
        skipFrames--;
        return {
            status:   "buffering",
            progress: 0,
            total:    SEQUENCE_LENGTH
        };
    }

    // FIX BUG 2: Không có tay → debounce reset thay vì reset ngay.
    if (!landmarks) {
        noHandCount++;
        if (noHandCount >= NO_HAND_RESET_FRAMES && frameBuffer.length > 0) {
            resetGRUBuffer();
        }
        return {
            status:   "buffering",
            progress: frameBuffer.length,
            total:    SEQUENCE_LENGTH
        };
    }

    // Có tay → reset counter vắng mặt
    noHandCount = 0;

    const keypoints = landmarks.flatMap(lm => [lm.x, lm.y, lm.z]); 
    frameBuffer.push(keypoints);

    if (frameBuffer.length < SEQUENCE_LENGTH) {
        return {
            status:   "buffering",
            progress: frameBuffer.length,
            total:    SEQUENCE_LENGTH
        };
    }

    if (frameBuffer.length > SEQUENCE_LENGTH) {
        frameBuffer = frameBuffer.slice(-SEQUENCE_LENGTH);
    }

    // ─── INFERENCE ───────────────────────────────────────────────────────────
    const flat   = frameBuffer.flat();
    const tensor = new ort.Tensor("float32", Float32Array.from(flat), [1, SEQUENCE_LENGTH, 63]);

    let results;
    try {
        results = await gruSession.run({ keypoints: tensor });
    } catch (e) {
        console.error("[GRU] ❌ Inference error:", e);
        return null;
    }

    const logits   = results.logits.data;              
    const expArr   = Array.from(logits).map(v => Math.exp(v));
    const sumExp   = expArr.reduce((a, b) => a + b, 0);
    const probs    = expArr.map(v => v / sumExp);

    const predIdx  = probs.indexOf(Math.max(...probs));
    const predProb = probs[predIdx];
    const label    = ACTIONS[predIdx] ?? "unknown";

    // 🛑 TRẢ KẾT QUẢ VÀ DỪNG Ở ĐÂY. KHÔNG ĐƯỢC RESET HAY SLICE BUFFER TẠI ĐÂY!
    if (predProb >= THRESHOLD) {
        return {
            status:     "predicted",
            label,
            confidence: Math.round(predProb * 100),
        };
    }

    return {
        status:     "low_confidence",
        label,
        confidence: Math.round(predProb * 100),
    };
}

// ─── RESET BUFFER (không skip) ───────────────────────────────────────────────
// Dùng khi khởi động session hoặc thoát luyện tập — không cần delay skip.
export function resetGRUBuffer() {
    console.log("[GRU] 🔄 Buffer reset");
    frameBuffer  = [];
    noHandCount  = 0;
    skipFrames   = 0;
}

// ─── RESET BUFFER + SKIP N FRAME ─────────────────────────────────────────────
// Dùng sau mỗi lần nhận diện (đúng hoặc sai) để counter giữ ở 0 trong ~N×33ms.
// Mặc định 15 frame ≈ 500ms — đủ để người dùng thấy rõ counter đã về 0.
export function resetGRUBufferAndSkip(n = 15) {
    console.log(`[GRU] 🔄 Buffer reset + skip ${n} frames`);
    frameBuffer  = [];
    noHandCount  = 0;
    skipFrames   = n;
}