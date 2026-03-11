// OcularTrace — main.js
// Gaze engine: original logic verbatim (main_slow.js)
// Reading tracker: additive layer only

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver } = vision;

// ── DOM ──────────────────────────────────────────────────────
const video          = document.getElementById('video');
const canvas         = document.getElementById('canvas');
const ctx            = canvas.getContext('2d');
const gazeDot        = document.getElementById('gaze-dot');
const calOverlay     = document.getElementById('calibration-overlay');
const calIntro       = document.getElementById('cal-intro');
const calPoint       = document.getElementById('cal-point');
const calStatus      = document.getElementById('cal-status');
const calProgress    = document.getElementById('cal-progress');
const calDots        = document.getElementById('cal-dots');
const calBeginBtn    = document.getElementById('cal-begin-btn');
const statusText     = document.getElementById('status-text');
const statusLed      = document.getElementById('status-led');
const toggleVideoBtn = document.getElementById('toggle-video-btn');
const startCalBtn    = document.getElementById('start-cal-btn');
const clearCalBtn    = document.getElementById('clear-cal-btn');
const resetBtn       = document.getElementById('reset-reading-btn');
const heatmapBtn     = document.getElementById('heatmap-btn');
const heatmapCanvas  = document.getElementById('heatmap-canvas');
const hctx           = heatmapCanvas.getContext('2d');
const progressFill   = document.getElementById('progress-fill');
const progressLabel  = document.getElementById('progress-pct-label');
const metricWpm      = document.getElementById('metric-wpm');
const metricPct      = document.getElementById('metric-pct');
const metricFix      = document.getElementById('metric-fix');
const metricReg      = document.getElementById('metric-reg');

// ── APP STATE  (original) ─────────────────────────────────────
let state = {
    faceLandmarker:   null,
    modelLoaded:      false,
    isCameraOn:       false,
    isCalibrating:    false,
    animationFrameId: null,
    videoStream:      null,
    coeffs:           { x: null, y: null },
    lastPrediction:   { x: 0.5, y: 0.5 }
};

// ── CONSTANTS  (original) ─────────────────────────────────────
const RIGHT_IRIS            = [474, 475, 476, 477];
const LEFT_IRIS             = [469, 470, 471, 472];
const SMOOTHING_ALPHA       = 0.45;
const HOLD_DURATION_MS      = 2500;
const TRANSITION_DURATION_MS = 1000;
const MODEL_STORAGE_KEY     = 'ocular_v6';   // bumped — forces fresh calibration

// ── DOT DISPLAY SMOOTHER (visual only, does not affect reading tracker) ──────
// A second EMA pass purely for the rendered dot position — makes it glide
// instead of snapping, without adding lag to the word-hit logic.
const DOT_ALPHA = 0.18;
const dotDisplay = { x: 0.5, y: 0.5 };
const CALIBRATION_POINTS = generateCalibrationPoints(15);

function generateCalibrationPoints(totalPoints = 12, margin = 0.05) {
    const points = [];
    const aspectRatio = window.innerWidth / window.innerHeight;
    let ny = Math.max(2, Math.round(Math.sqrt(totalPoints / aspectRatio)));
    let nx = Math.max(2, Math.round(ny * aspectRatio));
    const span = 1 - 2 * margin;
    nx = Math.max(3, nx);
    ny = Math.max(3, ny);
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const x = margin + (i / (nx - 1)) * span;
            const y = margin + (j / (ny - 1)) * span;
            points.push({ x, y });
        }
    }
    return points;
}

// ── MEDIAPIPE SETUP  (original) ──────────────────────────────
async function setupFaceLandmarker() {
    setStatus('Loading model…', 'warn');
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );
        state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            },
            outputFaceBlendshapes: false,
            runningMode: 'VIDEO',
            numFaces: 1
        });
        state.modelLoaded = true;
        setStatus('Model loaded.', 'ok');
    } catch (error) {
        console.error(error);
        setStatus('Error loading model.', 'err');
    }
}

// ── CAMERA  (original) ───────────────────────────────────────
async function toggleCamera() {
    if (state.isCameraOn) stopCamera();
    else await startCamera();
    updateUI();
}

async function startCamera() {
    try {
        const constraints = {
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
        };
        state.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = state.videoStream;
        video.onloadedmetadata = () => {
            video.play();
            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            state.isCameraOn = true;
            updateUI();
            setStatus(state.coeffs.x ? 'Tracking active.' : 'Camera on — calibrate first.', state.coeffs.x ? 'ok' : 'warn');
            predictLoop();
        };
    } catch (error) {
        console.error(error);
        setStatus('Camera access denied.', 'err');
    }
}

function stopCamera() {
    if (state.videoStream) state.videoStream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    state.isCameraOn = false;
    cancelAnimationFrame(state.animationFrameId);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setStatus('Camera off.', '');
    updateUI();
}

// ── PREDICTION LOOP  (original) ──────────────────────────────
async function predictLoop() {
    if (!state.isCameraOn || !state.modelLoaded) return;
    try {
        const results = await state.faceLandmarker.detectForVideo(video, performance.now());
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const irisData  = getIrisData(landmarks);
            drawIrisPoints(irisData.leftCenter, irisData.rightCenter);
            if (state.coeffs.x && state.coeffs.y && !state.isCalibrating) {
                predictGaze(irisData);
            }
        }
    } catch (error) {
        console.error(error);
    }
    state.animationFrameId = requestAnimationFrame(predictLoop);
}

// ── IRIS DATA  (original) ────────────────────────────────────
function getIrisData(landmarks) {
    const avg = (points) => points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
    const rpts = RIGHT_IRIS.map(i => landmarks[i]);
    const lpts = LEFT_IRIS.map(i => landmarks[i]);
    const rightCenter = { x: avg(rpts).x / rpts.length, y: avg(rpts).y / rpts.length };
    const leftCenter  = { x: avg(lpts).x / lpts.length, y: avg(lpts).y / lpts.length };
    return { rightCenter, leftCenter };
}

function drawIrisPoints(lc, rc) {
    ctx.fillStyle = 'rgba(100,255,100,0.8)';
    ctx.beginPath();
    ctx.arc(lc.x * canvas.width, lc.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,100,100,0.8)';
    ctx.beginPath();
    ctx.arc(rc.x * canvas.width, rc.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
}

// ── GAZE PREDICTION  (original) ──────────────────────────────
function predictGaze(irisData) {
    const f  = getFeatures(irisData);
    let px   = Math.max(0, Math.min(1, dot(f, state.coeffs.x)));
    let py   = Math.max(0, Math.min(1, dot(f, state.coeffs.y)));

    // Original EMA — used by reading tracker for word hit-testing
    state.lastPrediction.x = state.lastPrediction.x * (1 - SMOOTHING_ALPHA) + px * SMOOTHING_ALPHA;
    state.lastPrediction.y = state.lastPrediction.y * (1 - SMOOTHING_ALPHA) + py * SMOOTHING_ALPHA;

    // Second-pass EMA for dot rendering only — glides smoothly, no tracker lag
    dotDisplay.x += DOT_ALPHA * (state.lastPrediction.x - dotDisplay.x);
    dotDisplay.y += DOT_ALPHA * (state.lastPrediction.y - dotDisplay.y);

    gazeDot.style.left = `${dotDisplay.x * window.innerWidth}px`;
    gazeDot.style.top  = `${dotDisplay.y * window.innerHeight}px`;

    // Reading tracker gets the first-EMA coords (accurate but not over-smoothed)
    updateReading(state.lastPrediction.x, state.lastPrediction.y);

    // Tint dot teal while dwelling on an already-fixated word
    const overFixed = candidateIdx !== -1 && words[candidateIdx]?.fixated;
    if (overFixed) gazeDot.classList.add('fixation');
    else           gazeDot.classList.remove('fixation');
}

// ── FEATURES + MATHS  (original) ─────────────────────────────
function getFeatures({ rightCenter, leftCenter }) {
    const cx = (rightCenter.x + leftCenter.x) / 2;
    const cy = (rightCenter.y + leftCenter.y) / 2;
    return [1, cx, cy, cx*cx, cy*cy, cx*cy, cx*cx*cx, cy*cy*cy, cx*cx*cy, cx*cy*cy];
}

function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function transpose(A) { return A[0].map((_, i) => A.map(row => row[i])); }
function matMul(A, B) {
    const m = A.length, p = A[0].length, n = B[0].length;
    const C = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
        for (let k = 0; k < p; k++) {
            const v = A[i][k];
            for (let j = 0; j < n; j++) C[i][j] += v * B[k][j];
        }
    return C;
}
function solveLinear(Aorig, borig) {
    const n = Aorig.length, M = Aorig.map(r => r.slice()), b = borig.slice();
    for (let i = 0; i < n; i++) {
        let p = i;
        for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        [M[i], M[p]] = [M[p], M[i]]; [b[i], b[p]] = [b[p], b[i]];
        if (Math.abs(M[i][i]) < 1e-12) M[i][i] += 1e-8;
        const inv = 1 / M[i][i];
        for (let j = i; j < n; j++) M[i][j] *= inv;
        b[i] *= inv;
        for (let r = 0; r < n; r++) if (r !== i) {
            const f = M[r][i];
            for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
            b[r] -= f * b[i];
        }
    }
    return b;
}
function fitLS(Arows, targets) {
    const A = Arows, At = transpose(A), AtA = matMul(At, A);
    const Aty = matMul(At, targets.map(v => [v])), rhs = Aty.map(r => r[0]);
    return solveLinear(AtA, rhs);
}

// ── OUTLIER REMOVAL  (original) ──────────────────────────────
function removeOutliers(samples) {
    if (samples.length < 15) return samples;
    const points = samples.map(s => ({
        x: (s.iris.rightCenter.x + s.iris.leftCenter.x) / 2,
        y: (s.iris.rightCenter.y + s.iris.leftCenter.y) / 2
    }));
    const sortedX = points.map(p => p.x).sort((a, b) => a - b);
    const sortedY = points.map(p => p.y).sort((a, b) => a - b);
    const mid = Math.floor(sortedX.length / 2);
    const medianX = sortedX.length % 2 === 0 ? (sortedX[mid-1]+sortedX[mid])/2 : sortedX[mid];
    const medianY = sortedY.length % 2 === 0 ? (sortedY[mid-1]+sortedY[mid])/2 : sortedY[mid];
    const distances = points.map(p => Math.sqrt((p.x-medianX)**2 + (p.y-medianY)**2));
    const sortedD = [...distances].sort((a,b) => a-b);
    const medianD = sortedD.length % 2 === 0 ? (sortedD[mid-1]+sortedD[mid])/2 : sortedD[mid];
    const threshold = medianD * 3.0 + 1e-6;
    return samples.filter((_, i) => distances[i] < threshold);
}

// ── CALIBRATION  (original flow, adapted to new overlay UI) ──
function startCalibration() {
    state.isCalibrating = true;
    updateUI();
    calOverlay.classList.add('active');
    calIntro.classList.remove('hidden');
    calStatus.style.display  = 'none';
    calProgress.style.display = 'none';
}

calBeginBtn.addEventListener('click', async () => {
    calIntro.classList.add('hidden');
    await runCalibration();
});

async function runCalibration() {
    // Build progress dots
    calDots.innerHTML = '';
    CALIBRATION_POINTS.forEach((_, i) => {
        const d = document.createElement('div');
        d.className = 'cdot';
        d.id = `cdot-${i}`;
        calDots.appendChild(d);
    });
    calProgress.style.display = 'flex';
    calStatus.style.display   = 'block';

    // Show cal point element
    calPoint.style.display = 'block';

    let collectedData = [];
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);
    calStatus.textContent = 'Get ready…';
    await sleep(2000);

    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
        document.getElementById(`cdot-${i}`).classList.add('cur');
        const point = CALIBRATION_POINTS[i];

        // Original: animate transition then collect
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        calStatus.textContent = `Hold your gaze (${i + 1}/${CALIBRATION_POINTS.length})`;
        await sleep(200);

        calPoint.classList.add('collecting');
        const rawSamples      = await collectDataAtPoint(point);
        const filteredSamples = removeOutliers(rawSamples);
        collectedData.push(...filteredSamples);
        calPoint.classList.remove('collecting');

        document.getElementById(`cdot-${i}`).classList.remove('cur');
        document.getElementById(`cdot-${i}`).classList.add('done');
        lastPoint = point;
    }

    calStatus.textContent = 'Computing model…';

    if (collectedData.length > 50) {
        const features = collectedData.map(d => getFeatures(d.iris));
        const targetsX  = collectedData.map(d => d.target.x);
        const targetsY  = collectedData.map(d => d.target.y);
        state.coeffs.x  = fitLS(features, targetsX);
        state.coeffs.y  = fitLS(features, targetsY);
        saveCalibrationModel();
        setStatus('Calibration complete — tracking active.', 'ok');
    } else {
        setStatus(`Calibration failed — not enough data (${collectedData.length} samples).`, 'err');
    }

    calPoint.style.display    = 'none';
    calStatus.style.display   = 'none';
    calProgress.style.display = 'none';
    calOverlay.classList.remove('active');
    state.isCalibrating = false;
    updateUI();
}

// Original collectDataAtPoint — stores { iris, target } objects
async function collectDataAtPoint(point) {
    const samples = [];
    const endTime = performance.now() + HOLD_DURATION_MS;
    while (performance.now() < endTime) {
        const results = await state.faceLandmarker.detectForVideo(video, performance.now());
        if (results.faceLandmarks.length > 0) {
            samples.push({
                iris:   getIrisData(results.faceLandmarks[0]),
                target: point
            });
        }
        await sleep(1000 / 30);
    }
    return samples;
}

function placeCalPoint(normX, normY) {
    calPoint.style.left = `${normX * 100}%`;
    calPoint.style.top  = `${normY * 100}%`;
}

async function animatePointTransition(from, to, duration) {
    calStatus.textContent = 'Follow the point';
    const start = performance.now();
    return new Promise(resolve => {
        function frame() {
            const elapsed  = performance.now() - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased    = progress * (2 - progress);
            placeCalPoint(from.x + (to.x - from.x) * eased, from.y + (to.y - from.y) * eased);
            progress < 1 ? requestAnimationFrame(frame) : resolve();
        }
        requestAnimationFrame(frame);
    });
}

// ── LOCAL STORAGE  (original) ────────────────────────────────
function saveCalibrationModel() {
    localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(state.coeffs));
}
function loadCalibrationModel() {
    try {
        const data = JSON.parse(localStorage.getItem(MODEL_STORAGE_KEY));
        if (data && data.x && data.y) {
            state.coeffs = data;
            setStatus('Loaded saved calibration.', 'ok');
            return true;
        }
    } catch (e) {}
    setStatus('No calibration found.', 'warn');
    return false;
}
function clearCalibrationModel() {
    localStorage.removeItem(MODEL_STORAGE_KEY);
    state.coeffs = { x: null, y: null };
    setStatus('Calibration cleared.', 'warn');
    updateUI();
}

// ═══════════════════════════════════════════════════════════════
// READING TRACKER  — additive only, does not touch gaze engine
// ═══════════════════════════════════════════════════════════════

let words            = [];
let totalFixations   = 0;
let totalRegressions = 0;
let sessionStart     = null;

// -----------------------------------------------------------
// The tracker works with two concepts:
//
//  candidateIdx  — word the gaze is *currently hovering* on
//                  (shown in yellow immediately, no commitment yet)
//
//  committedIdx  — word we have *committed* to after DWELL_MS
//                  of continuous gaze. Regressions are only
//                  counted when committed word changes backward.
//
// This eliminates spurious regressions from normal gaze jitter.
// -----------------------------------------------------------
let candidateIdx   = -1;   // word currently under cursor (raw)
let candidateStart = null; // when we first entered this candidate
let committedIdx   = -1;   // last word we fully dwelled on

// Tune these two values to taste:
const DWELL_MS   = 120;   // ms gaze must stay on a word to "commit" to it
const FIXATE_MS  = 250;   // ms to mark word as fixated/read (must be >= DWELL_MS)

function collectWordBoxes() {
    words = [...document.querySelectorAll('.word')].map((el, i) => ({
        el, index: i,
        rect: el.getBoundingClientRect(),
        read: false, fixated: false
    }));
}
window.addEventListener('resize', collectWordBoxes);

function wordAtGaze(normX, normY) {
    const px = normX * window.innerWidth;
    const py = normY * window.innerHeight;
    for (const w of words) {
        // small hit-area expansion for calibration imprecision
        if (px >= w.rect.left - 10 && px <= w.rect.right  + 10 &&
            py >= w.rect.top  - 10 && py <= w.rect.bottom + 10) {
            return w.index;
        }
    }
    return -1;
}

function updateReading(nx, ny) {
    const idx = wordAtGaze(nx, ny);
    const now = performance.now();

    // ── Gaze moved to a different word (or off-text) ──────────
    if (idx !== candidateIdx) {
        // restore previous candidate to its permanent visual state
        if (candidateIdx !== -1) applyWordState(candidateIdx);

        candidateIdx   = idx;
        candidateStart = idx === -1 ? null : now;

        // show immediate hover highlight (yellow) but no commitment yet
        if (idx !== -1 && !words[idx].fixated) setWordClass(idx, 'gaze-active');
        return;
    }

    // ── Still on same word ────────────────────────────────────
    if (idx === -1 || candidateStart === null) return;

    const held = now - candidateStart;

    // Commit: after DWELL_MS continuously on this word, lock it in
    if (held >= DWELL_MS && idx !== committedIdx) {
        // Regression check — only against fully committed words
        if (committedIdx !== -1 && idx < committedIdx - 1 && words[committedIdx].fixated) {
            totalRegressions++;
            updateMetrics();
        }
        committedIdx = idx;
    }

    // Fixate: after FIXATE_MS, mark word as read
    if (held >= FIXATE_MS && !words[idx].fixated) {
        words[idx].fixated = true;
        words[idx].read    = true;
        totalFixations++;
        sessionStart = sessionStart ?? Date.now();
        heatmapPoints.push({ x: nx, y: ny });
        updateMetrics();
        setWordClass(idx, 'fixated');
        return;
    }

    // Still dwelling but not yet fixated — keep yellow
    if (!words[idx].fixated) setWordClass(idx, 'gaze-active');
}

function applyWordState(idx) {
    if (idx < 0 || idx >= words.length) return;
    const w = words[idx];
    if (w.fixated)   setWordClass(idx, 'fixated');
    else if (w.read) setWordClass(idx, 'read');
    else             setWordClass(idx, '');
}

function setWordClass(idx, cls) {
    if (idx < 0 || idx >= words.length) return;
    const el = words[idx].el;
    el.classList.remove('gaze-active', 'fixated', 'read');
    if (cls) el.classList.add(cls);
}

function updateMetrics() {
    const readCount = words.filter(w => w.read).length;
    const pct       = words.length ? Math.round((readCount / words.length) * 100) : 0;
    progressFill.style.width  = pct + '%';
    progressLabel.textContent = pct + '%';
    metricPct.textContent     = pct + '%';
    metricFix.textContent     = totalFixations;
    metricReg.textContent     = totalRegressions;
    if (sessionStart !== null && readCount > 1) {
        const elapsed = Date.now() - sessionStart;   // ms since first fixation
        const mins    = elapsed / 60000;
        metricWpm.textContent = mins > 0.05 ? Math.round(readCount / mins) : '—';
    }
}

function resetSession() {
    words.forEach((_, i) => setWordClass(i, ''));
    words.forEach(w => { w.read = false; w.fixated = false; });
    candidateIdx = -1; candidateStart = null; committedIdx = -1;
    totalFixations = 0; totalRegressions = 0; sessionStart = null;
    heatmapPoints = [];
    progressFill.style.width = '0%';
    progressLabel.textContent = '0%';
    metricWpm.textContent = '—';
    metricPct.textContent = '0%';
    metricFix.textContent = '0';
    metricReg.textContent = '0';
}

// ── HEATMAP ──────────────────────────────────────────────────
let heatmapPoints = [];
let showHeatmap   = false;
let heatmapRafId  = null;

function resizeHeatmap() {
    heatmapCanvas.width  = window.innerWidth;
    heatmapCanvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeHeatmap);

function drawHeatmap() {
    hctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
    for (const pt of heatmapPoints) {
        const x = pt.x * heatmapCanvas.width;
        const y = pt.y * heatmapCanvas.height;
        const r = 44;
        const g = hctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0,   'rgba(8,145,178,0.14)');
        g.addColorStop(0.5, 'rgba(8,145,178,0.06)');
        g.addColorStop(1,   'rgba(8,145,178,0)');
        hctx.beginPath(); hctx.arc(x, y, r, 0, 2*Math.PI);
        hctx.fillStyle = g; hctx.fill();
    }
    if (showHeatmap) heatmapRafId = requestAnimationFrame(drawHeatmap);
}

heatmapBtn.addEventListener('click', () => {
    showHeatmap = !showHeatmap;
    heatmapCanvas.classList.toggle('visible', showHeatmap);
    heatmapBtn.textContent = showHeatmap ? '⬡ Hide Map' : '⬡ Heatmap';
    heatmapBtn.classList.toggle('active', showHeatmap);
    if (showHeatmap) drawHeatmap();
    else { cancelAnimationFrame(heatmapRafId); hctx.clearRect(0,0,heatmapCanvas.width,heatmapCanvas.height); }
});

// ── UI HELPERS ────────────────────────────────────────────────
function setStatus(msg, level = '') {
    statusText.textContent   = msg;
    statusLed.className      = 'sled' + (level ? ' ' + level : '');
}

function updateUI() {
    toggleVideoBtn.textContent = state.isCameraOn ? 'Stop Camera' : 'Start Camera';
    startCalBtn.disabled       = !state.isCameraOn || state.isCalibrating;
    const has                  = !!(state.coeffs.x && state.coeffs.y);
    gazeDot.style.display      = has && state.isCameraOn && !state.isCalibrating ? 'block' : 'none';
    heatmapBtn.disabled        = !has;
}

function addListeners() {
    toggleVideoBtn.addEventListener('click', toggleCamera);
    startCalBtn.addEventListener('click', startCalibration);
    clearCalBtn.addEventListener('click', clearCalibrationModel);
    resetBtn.addEventListener('click', resetSession);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── INIT ─────────────────────────────────────────────────────
async function initialize() {
    resizeHeatmap();
    await setupFaceLandmarker();
    loadCalibrationModel();
    addListeners();
    updateUI();
    setTimeout(collectWordBoxes, 600);
}

initialize();
