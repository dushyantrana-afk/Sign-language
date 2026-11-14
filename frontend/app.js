// SignBridge AI — frontend logic (hackathon MVP)
//
// Hand-landmark detection runs HERE in the browser by default, via the
// MediaPipe Hands JS solution loaded in index.html, turning each frame
// into landmark JSON ({"hands": [{"label": "Right", "landmarks": [[x,y,z],
// ...21]}]}) sent to /api/enroll and /api/predict -- no video leaves the
// device in this mode. The "Detect hands on the server" checkbox switches
// to sending a raw frame ({"image": "data:..."}) instead, for the optional
// server-side detection path in backend/app/landmarks.py -- see that
// module's docstring and currentCapturePayload() below.
//
// Text<->speech uses the browser's own Web Speech API; no audio is ever
// sent to the server, in either detection mode.

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const startCameraBtn = document.getElementById("startCamera");
const stopCameraBtn = document.getElementById("stopCamera");
const cameraHint = document.getElementById("cameraHint");

let mediaStream = null;
let mpCamera = null; // MediaPipe's Camera helper (wraps getUserMedia + a render loop)
let handsModel = null;
let latestHandsResult = null; // most recent MediaPipe Hands onResults() payload
let handsModelLoadFailed = false;

const useServerDetectionCheckbox = document.getElementById("useServerDetection");
const detectionModeHint = document.getElementById("detectionModeHint");

// Check once at load whether the backend can even attempt server-side
// detection (it may still fail per-request, e.g. no internet to fetch the
// model -- /api/predict and /api/enroll report that per-call).
apiGet("/api/capabilities").then((caps) => {
  if (!caps.server_side_detection_importable) {
    useServerDetectionCheckbox.disabled = true;
    detectionModeHint.textContent =
      "This backend's mediapipe install doesn't support server-side detection — browser-side only.";
  }
});

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- MediaPipe Hands setup (browser-side detection) ----------
function ensureHandsModel() {
  if (handsModel) return true;
  if (typeof Hands === "undefined") {
    handsModelLoadFailed = true;
    return false;
  }
  try {
    handsModel = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    handsModel.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    handsModel.onResults(onHandsResults);
    return true;
  } catch (err) {
    handsModelLoadFailed = true;
    handsModel = null;
    return false;
  }
}

function onHandsResults(results) {
  latestHandsResult = results;
  drawOverlay(results);
}

function drawOverlay(results) {
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
  overlayCtx.save();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (results.multiHandLandmarks && typeof drawConnectors !== "undefined") {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(overlayCtx, landmarks, HAND_CONNECTIONS, { color: "#CADCFC", lineWidth: 2 });
      drawLandmarks(overlayCtx, landmarks, { color: "#1E2761", radius: 3 });
    }
  }
  overlayCtx.restore();
}

// ---------- Camera (starts only when the user chooses) ----------
async function startCamera() {
  try {
    const browserDetectionReady = ensureHandsModel();
    if (!browserDetectionReady && !useServerDetectionCheckbox.checked) {
      if (!useServerDetectionCheckbox.disabled) {
        // Hand-tracking library didn't load (e.g. CDN unreachable) but the
        // backend can do server-side detection -- fall back automatically
        // instead of just failing.
        useServerDetectionCheckbox.checked = true;
        detectionModeHint.textContent =
          "Browser hand-tracking library failed to load — switched to server-side detection automatically.";
      } else {
        cameraHint.textContent =
          "Hand-tracking library failed to load (check your internet connection) and this backend has no server-side fallback available — camera cannot start.";
        return;
      }
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = mediaStream;
    await video.play();

    if (browserDetectionReady) {
      mpCamera = new Camera(video, {
        onFrame: async () => {
          await handsModel.send({ image: video });
        },
        width: 640,
        height: 480,
      });
      mpCamera.start();
    } else {
      // No in-browser model running: just keep the <video> playing so
      // captureFrameDataUrl() has frames to grab for server-side mode.
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    }

    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    document.getElementById("captureEnroll").disabled = false;
    document.getElementById("toggleRecognition").disabled = false;
    cameraHint.textContent = "Camera is on. Turn it off any time with 'Stop camera'.";
  } catch (err) {
    cameraHint.textContent = "Could not access the camera: " + err.message;
  }
}

function captureFrameDataUrl(quality = 0.7) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

/** The payload to send to /api/enroll or /api/predict for the current
 * frame, honoring the detection-mode toggle. Returns null if there's
 * nothing to send yet (browser mode with no detection result so far). */
function currentCapturePayload() {
  if (useServerDetectionCheckbox.checked) {
    return { image: captureFrameDataUrl() };
  }
  return { hands: handsPayloadFromLatestResult() };
}

function stopCamera() {
  if (mpCamera) {
    mpCamera.stop();
    mpCamera = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  latestHandsResult = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  document.getElementById("captureEnroll").disabled = true;
  document.getElementById("toggleRecognition").disabled = true;
  stopRecognitionLoop();
  cameraHint.textContent = "Camera is off.";
}

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);

// ---------- Convert MediaPipe's latest result into the backend's JSON shape ----------
function handsPayloadFromLatestResult() {
  if (!latestHandsResult || !latestHandsResult.multiHandLandmarks) return [];
  const handedness = latestHandsResult.multiHandedness || [];
  return latestHandsResult.multiHandLandmarks.map((landmarks, i) => ({
    label: handedness[i] ? handedness[i].label : "Unknown",
    landmarks: landmarks.map((p) => [p.x, p.y, p.z]),
  }));
}

// ---------- API helpers ----------
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}
async function apiGet(path) {
  const res = await fetch(path);
  return res.json();
}

// ---------- Enrollment ----------
const enrollLabelInput = document.getElementById("enrollLabel");
const captureEnrollBtn = document.getElementById("captureEnroll");
const enrollStatus = document.getElementById("enrollStatus");
const vocabTableBody = document.getElementById("vocabTableBody");

async function refreshVocabulary() {
  const data = await apiGet("/api/vocabulary");
  vocabTableBody.innerHTML = "";
  (data.labels || []).forEach((label) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${data.example_counts[label]}</td>`;
    vocabTableBody.appendChild(tr);
  });
  document.getElementById("toggleRecognition").disabled = !data.ready_for_recognition || !mediaStream;
  return data;
}

captureEnrollBtn.addEventListener("click", async () => {
  const label = enrollLabelInput.value.trim();
  if (!label) {
    enrollStatus.textContent = "Enter a label first (e.g. 'hello').";
    return;
  }
  const payload = currentCapturePayload();
  if (payload.hands && payload.hands.length === 0) {
    enrollStatus.textContent = "No hand detected — make sure your hand is visible and well lit.";
    return;
  }
  enrollStatus.textContent = "Capturing…";
  const result = await apiPost("/api/enroll", { label, ...payload });
  if (result.error) {
    enrollStatus.textContent = "Error: " + result.error;
    return;
  }
  enrollStatus.textContent = `Captured "${label}" (${result.examples_for_label} example(s) so far).`;
  refreshVocabulary();
});

document.getElementById("refreshVocab").addEventListener("click", refreshVocabulary);

document.getElementById("resetAll").addEventListener("click", async () => {
  if (!confirm("Delete all enrolled signs and feedback? This cannot be undone.")) return;
  await apiPost("/api/reset", {});
  enrollStatus.textContent = "All enrolled data deleted.";
  refreshVocabulary();
});

// ---------- Live recognition ----------
const toggleRecognitionBtn = document.getElementById("toggleRecognition");
const predictionBox = document.getElementById("predictionBox");
const predictionLabelEl = predictionBox.querySelector(".prediction-label");
const predictionMetaEl = predictionBox.querySelector(".prediction-meta");
const confirmBox = document.getElementById("confirmBox");
const confirmLabelEl = document.getElementById("confirmLabel");
const correctPicker = document.getElementById("correctPicker");
const speakPredictionCheckbox = document.getElementById("speakPrediction");

let recognizing = false;
let recognitionTimer = null;
let lastSpokenLabel = null;

function setPredictionDisplay(state, label, meta) {
  predictionBox.classList.remove("idle", "ok", "low");
  predictionBox.classList.add(state);
  predictionLabelEl.textContent = label;
  predictionMetaEl.textContent = meta;
}

async function recognitionTick() {
  if (!recognizing) return;
  const payload = currentCapturePayload();
  try {
    const result = await apiPost("/api/predict", payload);
    if (result.reason === "no_hand_detected") {
      setPredictionDisplay("idle", "—", "No hand detected");
      confirmBox.classList.add("hidden");
    } else if (result.reason === "not_trained") {
      setPredictionDisplay("idle", "—", "Enroll at least 2 signs with 3 examples each first");
    } else if (result.needs_confirmation) {
      setPredictionDisplay(
        "low",
        result.label || "Uncertain",
        result.label
          ? `Confidence ${(result.confidence * 100).toFixed(0)}% — please confirm`
          : "Gesture doesn't match enrolled signs closely enough"
      );
      if (result.label) showConfirmation(result);
    } else {
      setPredictionDisplay("ok", result.label, `Confidence ${(result.confidence * 100).toFixed(0)}%`);
      confirmBox.classList.add("hidden");
      if (speakPredictionCheckbox.checked && result.label !== lastSpokenLabel) {
        speak(result.label);
        lastSpokenLabel = result.label;
      }
    }
  } catch (err) {
    setPredictionDisplay("idle", "—", "Prediction error: " + err.message);
  }
  recognitionTimer = setTimeout(recognitionTick, 900);
}

function showConfirmation(result) {
  confirmBox.classList.remove("hidden");
  confirmLabelEl.textContent = result.label;
  correctPicker.classList.add("hidden");
  correctPicker.innerHTML = "";

  const consentChecked = () => document.getElementById("feedbackConsent").checked;

  document.getElementById("confirmYes").onclick = async () => {
    confirmBox.classList.add("hidden");
    if (speakPredictionCheckbox.checked) speak(result.label);
    if (consentChecked()) {
      await apiPost("/api/feedback", {
        predicted_label: result.label,
        correct_label: result.label,
        confidence: result.confidence,
        consent: true,
      });
    }
  };

  document.getElementById("confirmNo").onclick = async () => {
    const vocab = await apiGet("/api/vocabulary");
    correctPicker.classList.remove("hidden");
    correctPicker.innerHTML = "";
    (vocab.labels || []).forEach((label) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = label;
      b.onclick = async () => {
        confirmBox.classList.add("hidden");
        if (consentChecked()) {
          await apiPost("/api/feedback", {
            predicted_label: result.label,
            correct_label: label,
            confidence: result.confidence,
            consent: true,
          });
        }
      };
      correctPicker.appendChild(b);
    });
  };
}

function startRecognitionLoop() {
  recognizing = true;
  toggleRecognitionBtn.textContent = "Stop recognizing";
  recognitionTick();
}
function stopRecognitionLoop() {
  recognizing = false;
  clearTimeout(recognitionTimer);
  toggleRecognitionBtn.textContent = "Start recognizing";
  setPredictionDisplay("idle", "—", "");
}

toggleRecognitionBtn.addEventListener("click", () => {
  if (recognizing) stopRecognitionLoop();
  else startRecognitionLoop();
});

// ---------- Phrase shortcuts ----------
document.querySelectorAll(".chip[data-phrase]").forEach((chip) => {
  chip.addEventListener("click", () => speak(chip.dataset.phrase));
});

// ---------- Text-to-speech (browser SpeechSynthesis) ----------
function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(utter);
}

document.getElementById("speakText").addEventListener("click", () => {
  const text = document.getElementById("ttsText").value.trim();
  if (text) speak(text);
});

// ---------- Speech-to-text / captions (browser SpeechRecognition) ----------
const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let sttInstance = null;
let listening = false;
const captionBox = document.getElementById("captionBox");
const toggleSttBtn = document.getElementById("toggleStt");

if (!SpeechRecognitionImpl) {
  toggleSttBtn.disabled = true;
  captionBox.textContent = "Speech-to-text isn't supported in this browser.";
} else {
  sttInstance = new SpeechRecognitionImpl();
  sttInstance.continuous = true;
  sttInstance.interimResults = true;
  sttInstance.onresult = (event) => {
    let text = "";
    for (let i = 0; i < event.results.length; i++) {
      text += event.results[i][0].transcript + " ";
    }
    captionBox.textContent = text.trim();
  };
  sttInstance.onerror = (event) => {
    captionBox.textContent = "Speech recognition error: " + event.error;
  };
  sttInstance.onend = () => {
    if (listening) sttInstance.start(); // keep listening until user stops
  };
}

toggleSttBtn.addEventListener("click", () => {
  if (!sttInstance) return;
  listening = !listening;
  if (listening) {
    sttInstance.start();
    toggleSttBtn.textContent = "Stop listening";
  } else {
    sttInstance.stop();
    toggleSttBtn.textContent = "Start listening";
  }
});

// ---------- Init ----------
refreshVocabulary();
setPredictionDisplay("idle", "—", "Start the camera, then enroll or recognize");
