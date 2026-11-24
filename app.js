import { FilesetResolver, GestureRecognizer } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const toast = document.querySelector("#toast");

const slides = [
  {
    kicker: "GestureSlide · Live demo",
    title: "Let your hands<br>lead the way.",
    body: "Navigate a presentation naturally. The camera recognizes your hand and the controls respond right here in the browser.",
  },
  {
    kicker: "01 · Move naturally",
    title: "No clicker.<br>No interruption.",
    body: "A steady open palm moves forward. A closed fist takes you back. Keep your place and keep your eyes on the audience.",
  },
  {
    kicker: "02 · Designed for clarity",
    title: "Your camera.<br>Your control.",
    body: "The hand gesture model runs on this device. The live preview and landmarks stay in your browser session.",
  },
  {
    kicker: "03 · Your turn",
    title: "Try a gesture<br>to take control.",
    body: "Show an open palm to go forward, a fist to return, thumbs up to start or end, and a peace sign to blank the screen.",
  },
];
const connections = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];
let stream = null;
let recognizer = null;
let rafId = 0;
let runId = 0;
let slideIndex = 0;
let isPresenting = false;
let isBlackout = false;
let currentGesture = "";
let currentCandidate = "";
let candidateFrames = 0;
let lastFiredGesture = "";
let armed = true;
let neutralFrames = 0;
let lastActionAt = 0;
let lastInferenceAt = 0;
let lastVideoTime = -1;
let fpsFrames = 0;
let fpsAt = performance.now();
let toastTimer = 0;

function setCameraStatus(text, live = false) {
  document.querySelector("#cameraStatus").textContent = text;
  document.querySelector("#topStatus").textContent = live ? "Camera live" : text;
  document.querySelector("#cameraDot").classList.toggle("on", live);
  document.querySelector("#liveDot").classList.toggle("on", live);
}

function notify(message, error = false) {
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2500);
}

function renderSlide() {
  const slide = slides[slideIndex];
  document.querySelector("#slideKicker").textContent = slide.kicker;
  document.querySelector("#slideTitle").innerHTML = slide.title;
  document.querySelector("#slideBody").textContent = slide.body;
  document.querySelector("#slideNumber").textContent = `${String(slideIndex + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
  document.querySelector("#counter").textContent = `${slideIndex + 1} / ${slides.length}`;
  document.querySelector("#slide").classList.toggle("blackout", isBlackout);
  document.querySelector("#deckStatus").textContent = isBlackout ? "Screen blanked" : isPresenting ? "Presentation live" : "Ready to present";
  document.querySelector("#presentBtn").textContent = isPresenting ? "End show" : "Start show";
}

function goToSlide(nextIndex) {
  slideIndex = Math.max(0, Math.min(slides.length - 1, nextIndex));
  isBlackout = false;
  renderSlide();
}

function showGesture(name, score, handedness) {
  currentGesture = name;
  document.querySelector("#gestureName").textContent = name ? name.replaceAll("_", " ") : "No gesture";
  document.querySelector("#confidence").textContent = name ? `${Math.round(score * 100)}% confidence` : "—";
  document.querySelector("#handPill").textContent = name ? `Hand detected${handedness ? ` · ${handedness}` : ""}` : "No hand";
  document.querySelectorAll(".gesture-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.gesture === name);
  });
}

function fireGesture(name) {
  const now = performance.now();
  if (now - lastActionAt < 1050) return;
  lastActionAt = now;
  lastFiredGesture = name;
  armed = false;
  switch (name) {
    case "Open_Palm":
      if (slideIndex < slides.length - 1) {
        goToSlide(slideIndex + 1);
        notify("Next slide");
      } else notify("You’re on the last slide");
      break;
    case "Closed_Fist":
      if (slideIndex > 0) {
        goToSlide(slideIndex - 1);
        notify("Previous slide");
      } else notify("You’re on the first slide");
      break;
    case "Thumb_Up":
      isPresenting = !isPresenting;
      isBlackout = false;
      renderSlide();
      notify(isPresenting ? "Presentation started" : "Presentation ended");
      break;
    case "Victory":
      isBlackout = !isBlackout;
      renderSlide();
      notify(isBlackout ? "Screen blanked · show peace sign again to restore" : "Screen restored");
      break;
    default:
      armed = true;
  }
}

function drawLandmarks(landmarks) {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks?.length) return;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const hand of landmarks) {
    ctx.strokeStyle = "rgba(114,231,237,.92)";
    ctx.lineWidth = Math.max(3, canvas.width / 280);
    for (const [a, b] of connections) {
      const p = hand[a], q = hand[b];
      if (!p || !q) continue;
      ctx.beginPath();
      ctx.moveTo(p.x * canvas.width, p.y * canvas.height);
      ctx.lineTo(q.x * canvas.width, q.y * canvas.height);
      ctx.stroke();
    }
    for (const point of hand) {
      ctx.beginPath();
      ctx.fillStyle = "#e8ffff";
      ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(3, canvas.width / 230), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function updateCandidate(name) {
  if (!name) {
    candidateFrames = 0;
    currentCandidate = "";
    neutralFrames += 1;
    if (neutralFrames >= 4) {
      armed = true;
      lastFiredGesture = "";
    }
    return;
  }
  neutralFrames = 0;
  if (currentCandidate === name) candidateFrames += 1;
  else {
    currentCandidate = name;
    candidateFrames = 1;
  }
  if (candidateFrames >= 6 && (armed || name !== lastFiredGesture)) fireGesture(name);
}

function renderLoop(id) {
  if (id !== runId) return;
  rafId = requestAnimationFrame(() => renderLoop(id));
  if (document.hidden || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  const now = performance.now();
  if (now - lastInferenceAt < 65) return;
  lastVideoTime = video.currentTime;
  lastInferenceAt = now;
  try {
    const result = recognizer.recognizeForVideo(video, now);
    const categories = result.gestures?.[0] || [];
    const best = categories[0];
    const name = best && best.categoryName !== "None" && best.score >= 0.68 ? best.categoryName : "";
    const handedness = result.handednesses?.[0]?.[0]?.categoryName || "";
    showGesture(name, best?.score || 0, handedness);
    drawLandmarks(result.landmarks || []);
    updateCandidate(name);
    fpsFrames += 1;
    if (now - fpsAt >= 1000) {
      document.querySelector("#modelPill").textContent = `LIVE · ${fpsFrames} FPS`;
      fpsFrames = 0;
      fpsAt = now;
    }
  } catch (error) {
    console.error("Gesture inference failed", error);
    stopCamera("Recognition stopped");
    notify("Camera recognition stopped. Please try again.", true);
  }
}

async function createRecognizer() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  };
  try {
    return await GestureRecognizer.createFromOptions(vision, options);
  } catch (gpuError) {
    console.warn("GPU initialization failed; retrying on CPU", gpuError);
    return GestureRecognizer.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    });
  }
}

async function startCamera() {
  startBtn.disabled = true;
  document.querySelector("#cameraHelp").textContent = "Loading the on-device gesture model…";
  document.querySelector("#modelPill").textContent = "Loading AI model…";
  setCameraStatus("Loading model");
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access needs HTTPS or localhost in a supported browser.");
    }
    const id = ++runId;
    recognizer = await createRecognizer();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    });
    if (id !== runId) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    video.srcObject = stream;
    await video.play();
    document.querySelector("#placeholder").hidden = true;
    document.querySelector("#modelPill").textContent = "ON-DEVICE MODEL";
    document.querySelector("#cameraHelp").textContent = "Hold a gesture steady to trigger it once. Drop your hand to re-arm.";
    startBtn.textContent = "Camera running";
    stopBtn.disabled = false;
    setCameraStatus("Live", true);
    showGesture("", 0, "");
    renderLoop(id);
  } catch (error) {
    console.error("Could not start camera", error);
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    recognizer?.close();
    recognizer = null;
    const message = error?.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow camera access for this site, then try again."
      : error?.name === "NotFoundError"
        ? "No camera was found. Connect a webcam and try again."
        : error?.message || "Could not start the camera.";
    document.querySelector("#cameraHelp").textContent = message;
    document.querySelector("#modelPill").textContent = "CAMERA OFF";
    document.querySelector("#gestureName").textContent = "Camera unavailable";
    setCameraStatus("Camera unavailable");
    notify(message, true);
  } finally {
    startBtn.disabled = false;
  }
}

function stopCamera(status = "Camera off") {
  runId += 1;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.pause();
  video.srcObject = null;
  recognizer?.close();
  recognizer = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.querySelector("#placeholder").hidden = false;
  document.querySelector("#modelPill").textContent = "AI model ready to load";
  document.querySelector("#cameraHelp").textContent = "Use HTTPS or localhost and allow camera access in your browser.";
  document.querySelector("#gestureName").textContent = "Waiting for camera";
  document.querySelector("#confidence").textContent = "—";
  document.querySelector("#handPill").textContent = "No hand";
  document.querySelectorAll(".gesture-card").forEach((card) => card.classList.remove("active"));
  startBtn.textContent = "Start camera";
  stopBtn.disabled = true;
  setCameraStatus(status);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", () => stopCamera());
document.querySelector("#prevBtn").addEventListener("click", () => goToSlide(slideIndex - 1));
document.querySelector("#nextBtn").addEventListener("click", () => goToSlide(slideIndex + 1));
document.querySelector("#presentBtn").addEventListener("click", () => {
  isPresenting = !isPresenting;
  isBlackout = false;
  renderSlide();
  notify(isPresenting ? "Presentation started" : "Presentation ended");
});
document.addEventListener("keydown", (event) => {
  if (event.repeat || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.key === "ArrowRight" || event.key === " ") goToSlide(slideIndex + 1);
  if (event.key === "ArrowLeft") goToSlide(slideIndex - 1);
  if (event.key.toLowerCase() === "b") {
    isBlackout = !isBlackout;
    renderSlide();
  }
  if (event.key === "Escape" && isPresenting) {
    isPresenting = false;
    isBlackout = false;
    renderSlide();
  }
});
window.addEventListener("beforeunload", () => {
  stream?.getTracks().forEach((track) => track.stop());
  recognizer?.close();
});
renderSlide();
