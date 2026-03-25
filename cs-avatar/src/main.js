// loads Three.js, .glb avatar and allow zoom and rotation 
import "./style.css";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";

const scene = new THREE.Scene();

// load background cartoon 
const textureLoader = new THREE.TextureLoader();
textureLoader.load(
  "/Loughborough_Gemini_Cartoon.png",
  (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    scene.background = texture;
  },
  undefined,
  (err) => console.error("Background image load error:", err)
);

// perspective camera settings 
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.style.margin = "0";
document.body.appendChild(renderer.domElement);

// Orbit controls 
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;

// Lighting
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(2, 5, 2);
scene.add(dirLight);

// load avatar 
const loader = new GLTFLoader();
let mixer = null;

loader.load(
  "/male2.glb",
  (gltf) => {
    const avatar = gltf.scene;
    scene.add(avatar);

    // autot-center and auto-frame 
    const box = new THREE.Box3().setFromObject(avatar);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    avatar.position.sub(center); // center at origin

    // place feet on ground
    const boxGround = new THREE.Box3().setFromObject(avatar);
    avatar.position.y -= boxGround.min.y;

    const box2 = new THREE.Box3().setFromObject(avatar);
    const size2 = box2.getSize(new THREE.Vector3());
    const center2 = box2.getCenter(new THREE.Vector3());

    controls.target.copy(center2);
    controls.update();

    const maxDim = Math.max(size2.x, size2.y, size2.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs((maxDim / 2) / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(center2.x, center2.y + maxDim * 0.2, center2.z + cameraZ);
    camera.near = cameraZ / 100;
    camera.far = cameraZ * 100;
    camera.updateProjectionMatrix();

    controls.minDistance = cameraZ * 0.3;
    controls.maxDistance = cameraZ * 3;

    // animation controls 
    if (gltf.animations && gltf.animations.length > 0) {
      mixer = new THREE.AnimationMixer(avatar);
      const action = mixer.clipAction(gltf.animations[0]);
      action.play();
      console.log("Animations:", gltf.animations.map((a) => a.name));
    } else {
      console.log("No animations found in GLB.");
    }
  },
  undefined,
  (err) => console.error("GLB load error:", err)
);

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (mixer) mixer.update(dt);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const chatLog = document.getElementById("chatLog");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const statusEl = document.getElementById("status");
const DEFAULT_STATUS = "Type a question or press the microphone to speak.";
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");
const QUERY_ENDPOINT = `${API_BASE_URL}/query`;

let isSending = false;
let recognition = null;
let isListening = false;

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

const defaultLinkOpenRule =
  md.renderer.rules.link_open ||
  ((tokens, idx, options, env, self) =>
    self.renderToken(tokens, idx, options));

md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  token.attrSet("target", "_blank");
  token.attrSet("rel", "noopener noreferrer");
  return defaultLinkOpenRule(tokens, idx, options, env, self);
};

function renderUserMessage(text) {
  const content = document.createElement("div");
  content.className = "msg-content";
  content.textContent = text;
  return content;
}

function renderAvatarMessage(markdownText) {
  const content = document.createElement("div");
  content.className = "msg-content markdown";

  const rawHtml = md.render(markdownText);
  const safeHtml = DOMPurify.sanitize(rawHtml, {
    USE_PROFILES: { html: true },
    ADD_ATTR: ["target", "rel"],
  });
  content.innerHTML = safeHtml;
  return content;
}

function appendMessage(role, text) {
  const message = document.createElement("article");
  message.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "me" ? "You" : "Avatar";
  message.appendChild(label);

  const content =
    role === "bot" ? renderAvatarMessage(text) : renderUserMessage(text);
  message.appendChild(content);

  chatLog.appendChild(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setRequestState(pending) {
  isSending = pending;
  sendBtn.disabled = pending;
}

async function handleSend(questionText) {
  const q = (questionText ?? textInput.value).trim();
  if (!q) {
    return;
  }
  if (isSending) {
    statusEl.textContent = "Please wait for the current answer.";
    return;
  }

  appendMessage("me", q);
  textInput.value = "";
  setRequestState(true);
  statusEl.textContent = "Avatar is thinking...";

  try {
    const response = await fetch(QUERY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : null;

    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && payload.detail
          ? String(payload.detail)
          : `Request failed with status ${response.status}`;
      throw new Error(detail);
    }

    const answer =
      payload && typeof payload.answer === "string" && payload.answer.trim()
        ? payload.answer.trim()
        : "I could not generate an answer from the backend.";
    appendMessage("bot", answer);
    statusEl.textContent = "Answer received. Ask another question any time.";
  } catch (error) {
    console.error("Chat request failed:", error);
    appendMessage(
      "bot",
      "Sorry, I could not reach the backend just now. Please try again."
    );
    const detail =
      error instanceof Error ? error.message : "Unknown network error.";
    statusEl.textContent = `Error: ${detail}`;
  } finally {
    setRequestState(false);
  }
}

sendBtn.addEventListener("click", () => handleSend());
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleSend();
});

statusEl.textContent = DEFAULT_STATUS;

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "en-GB";
  recognition.interimResults = false;
  recognition.continuous = false;

  recognition.onstart = () => {
    isListening = true;
    statusEl.textContent = "Listening… ask your question.";
    micBtn.textContent = "🛑";
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    statusEl.textContent = `Heard: "${transcript}"`;
    textInput.value = transcript;

    // Voice-first behavior: auto-send as soon as we get text
    handleSend(transcript);
  };

  recognition.onerror = (event) => {
    statusEl.textContent = `Mic error: ${event.error}. You can still type.`;
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = "🎤";
    // Don’t overwrite error messages; only reset if currently "Listening"
    if (statusEl.textContent.startsWith("Listening")) {
      statusEl.textContent = DEFAULT_STATUS;
    }
  };

  micBtn.addEventListener("click", () => {
    if (!recognition) return;
    if (isSending) {
      statusEl.textContent =
        "Please wait for the current answer before recording another question.";
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });
} else {
  // if browser is unable to support speech-to-text
  micBtn.disabled = true;
  micBtn.title = "Speech-to-text not supported in this browser.";
  statusEl.textContent =
    "Speech-to-text not supported here. Please type your question.";
}
