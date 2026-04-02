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
const avatars  = { idle: null, talking: null };
const mixers   = { idle: null, talking: null };
let mouthMesh = null;
let mouthOpenIndex = -1;
let isSpeaking = false;
let avatarHeadPosition = null;
let currentAvatarState = 'idle';
let idleAvatarPosition = null; // shared position applied to all avatars

function setAvatarState(state) {
  const next = avatars[state] ? state : 'idle';
  if (avatars[currentAvatarState]) avatars[currentAvatarState].visible = false;
  if (avatars[next])              avatars[next].visible = true;
  currentAvatarState = next;
}

// Centre and ground an avatar; returns its final bounding box
function positionAvatar(avatar) {
  const box = new THREE.Box3().setFromObject(avatar);
  avatar.position.sub(box.getCenter(new THREE.Vector3()));
  const boxGround = new THREE.Box3().setFromObject(avatar);
  avatar.position.y -= boxGround.min.y;
  return new THREE.Box3().setFromObject(avatar);
}

// Set up camera from the idle avatar's bounding box — called as soon as it loads
function setupCamera(box) {
  const size   = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  controls.target.copy(center);
  controls.update();

  avatarHeadPosition = new THREE.Vector3(center.x, box.max.y * 1.1, center.z);

  const maxDim  = Math.max(size.x, size.y, size.z);
  const fov     = camera.fov * (Math.PI / 180);
  const cameraZ = Math.abs((maxDim / 2) / Math.tan(fov / 2)) * 1.1;

  camera.position.set(center.x, center.y + maxDim * 0.35, center.z + cameraZ);
  camera.near = cameraZ / 100;
  camera.far  = cameraZ * 100;
  camera.updateProjectionMatrix();
  controls.minDistance = cameraZ * 0.3;
  controls.maxDistance = cameraZ * 3;
}

// Load idle avatar first — show it immediately and start the intro
loader.load("/male2.glb", (gltf) => {
  const avatar = gltf.scene;
  scene.add(avatar);
  const box = positionAvatar(avatar);
  idleAvatarPosition = avatar.position.clone();
  avatars.idle = avatar;
  // avatar.visible is true by default — shown straight away

  setupCamera(box);

  if (gltf.animations?.length) {
    mixers.idle = new THREE.AnimationMixer(avatar);
    mixers.idle.clipAction(gltf.animations[0]).play();
  }

  // Show intro; speak on first interaction (Chrome autoplay policy)
  const intro = "Hi, welcome to Loughborough University's Open Day! I'm your virtual assistant, here to help answer any questions you might have about our courses, campus, student life, or anything else. Feel free to type or speak your question to get started.";
  appendMessage("bot", intro);
  micBtn.disabled = true;
  micBtn.title = "Please wait for the introduction to finish.";
  let introSpoken = false;
  function speakIntroOnce() {
    if (introSpoken) return;
    introSpoken = true;
    document.removeEventListener("click", speakIntroOnce);
    document.removeEventListener("keydown", speakIntroOnce);
    const introUtterance = new SpeechSynthesisUtterance(intro.replace(/[#*_`~\[\]()>|\\-]/g, "").replace(/\n+/g, " ").trim());
    introUtterance.lang = "en-GB";
    introUtterance.rate = 1;
    introUtterance.pitch = 1;
    if (cachedVoice) introUtterance.voice = cachedVoice;
    introUtterance.onend   = () => { isSpeaking = false; setAvatarState('idle'); micBtn.disabled = false; micBtn.title = ""; };
    introUtterance.onerror = () => { isSpeaking = false; setAvatarState('idle'); micBtn.disabled = false; micBtn.title = ""; };
    isSpeaking = true;
    setAvatarState('talking');
    window.speechSynthesis.speak(introUtterance);
  }
  document.addEventListener("click", speakIntroOnce);
  document.addEventListener("keydown", speakIntroOnce);
}, undefined, (err) => console.error("idle GLB load error:", err));

// Load talking avatar in parallel — hides itself until needed
loader.load("/male2_talking.glb", (gltf) => {
  const avatar = gltf.scene;
  scene.add(avatar);
  positionAvatar(avatar);
  // Snap to the same position as the idle avatar so they perfectly overlap
  if (idleAvatarPosition) avatar.position.copy(idleAvatarPosition);
  avatar.visible = false;
  avatars.talking = avatar;

  avatar.traverse((child) => {
    if (child.isMesh && child.morphTargetDictionary && "mouthOpen" in child.morphTargetDictionary) {
      mouthMesh = child;
      mouthOpenIndex = child.morphTargetDictionary["mouthOpen"];
      console.log("Found mouthOpen morph target on:", child.name, "at index:", mouthOpenIndex);
    }
  });
  if (gltf.animations?.length) {
    mixers.talking = new THREE.AnimationMixer(avatar);
    mixers.talking.clipAction(gltf.animations[0]).play();
  }
}, undefined, (err) => console.error("talking GLB load error:", err));

const clock = new THREE.Clock();
let mouthTime = 0;
let mouthTarget = 0;
let mouthCurrent = 0;
let nextChangeTime = 0;
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  // tick all mixers so hidden avatars stay in sync
  if (mixers.idle)    mixers.idle.update(dt);
  if (mixers.talking) mixers.talking.update(dt);

  // animate mouth while speaking — varied rhythm for natural look
  if (mouthMesh && mouthOpenIndex >= 0) {
    if (isSpeaking) {
      mouthTime += dt;
      if (mouthTime >= nextChangeTime) {
        // randomly pick a new mouth openness target
        const isSilentGap = Math.random() < 0.15;
        mouthTarget = isSilentGap ? 0.05 : 0.15 + Math.random() * 0.55;
        // vary how long each position holds (fast syllables + brief pauses)
        nextChangeTime = mouthTime + 0.06 + Math.random() * 0.12;
      }
      // smooth interpolation toward target
      mouthCurrent += (mouthTarget - mouthCurrent) * Math.min(1, dt * 18);
      mouthMesh.morphTargetInfluences[mouthOpenIndex] = mouthCurrent;
    } else {
      mouthCurrent *= 0.85;
      mouthTarget = 0;
      mouthMesh.morphTargetInfluences[mouthOpenIndex] = mouthCurrent;
    }
  }

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
let pendingEditElements = null;
let lastQuestion = "";
let lastAnswer = "";

const CANNED_RESPONSES = [
  {
    patterns: [/^thank(s| you)/i, /^cheers/i],
    reply: "You're welcome! I hope I was able to help. Feel free to ask if you have any more questions."
  },
  {
    patterns: [/^hello/i, /^hi\b/i, /^hey\b/i, /^good (morning|afternoon|evening)/i],
    reply: "Hello! I'm the Loughborough University virtual assistant. How can I help you today?"
  },
  {
    patterns: [/^bye/i, /^goodbye/i, /^see you/i],
    reply: "Goodbye! I hope I was helpful. Feel free to come back if you have more questions about Loughborough University."
  },
  {
    patterns: [/^who are you/i, /^what are you/i],
    reply: "I'm a virtual assistant for Loughborough University, here to help answer questions about courses, open days, entry requirements, and student life."
  },
  {
    patterns: [/^what can you (help|do)/i, /^what do you know/i, /^what can (i|you) ask/i],
    reply: "I can answer questions about Loughborough University courses, entry requirements, open days, student life, campus facilities, and much more. Go ahead and ask!"
  },
  {
    patterns: [/^(that'?s?|it'?s?) (helpful|great|brilliant|perfect|amazing|awesome)/i, /^great answer/i],
    reply: "Glad I could help! Let me know if you have any more questions."
  },
  {
    patterns: [/^(can you )?repeat that/i, /^say that again/i, /^pardon/i],
    reply: () => lastAnswer || "I'm sorry, I don't have a previous answer to repeat."
  },
  {
    patterns: [/^(i don'?t understand|can you explain|i'?m? confused)/i],
    reply: "I'm sorry if that wasn't clear. Could you rephrase your question and I'll try to give a better answer."
  },
];

function checkCannedResponse(q) {
  for (const { patterns, reply } of CANNED_RESPONSES) {
    if (patterns.some(p => p.test(q.trim()))) {
      return typeof reply === "function" ? reply() : reply;
    }
  }
  return null;
}

function buildQuery(q) {
  if (/^(what about|and what about|how about|tell me more|more about|can you elaborate|what else)/i.test(q.trim()) && lastQuestion) {
    return `${q} (in relation to: ${lastQuestion})`;
  }
  return q;
}

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

function createFeedbackRow(text) {
  const feedbackRow = document.createElement("div");
  feedbackRow.className = "feedback-row";

  const label = document.createElement("span");
  label.className = "feedback-label";
  label.textContent = "Was your question answered?";

  const yesBtn = document.createElement("button");
  yesBtn.className = "feedback-btn feedback-yes";
  yesBtn.textContent = "Yes";

  const noBtn = document.createElement("button");
  noBtn.className = "feedback-btn feedback-no";
  noBtn.textContent = "No";

  function handleFeedback(answered) {
    yesBtn.disabled = true;
    noBtn.disabled = true;
    feedbackRow.innerHTML = answered
      ? `<span class="feedback-thanks">Glad we could help!</span>`
      : `<span class="feedback-thanks">Sorry about that \u2014 we'll try to improve.</span>`;
    console.log("Feedback:", answered ? "answered" : "not answered", "for:", text);
    // TODO: send feedback to your backend
    // fetch("/api/feedback", { method: "POST", body: JSON.stringify({ answer: text, resolved: answered }) });
  }

  yesBtn.addEventListener("click", () => handleFeedback(true));
  noBtn.addEventListener("click", () => handleFeedback(false));

  feedbackRow.appendChild(label);
  feedbackRow.appendChild(yesBtn);
  feedbackRow.appendChild(noBtn);
  return feedbackRow;
}

let cachedVoice = null;
function loadPreferredVoice() {
  const voices = window.speechSynthesis.getVoices();
  // Prefer local voices — they start instantly vs network voices which require a server round-trip
  cachedVoice =
    voices.find(v => v.localService && v.lang === "en-GB") ||
    voices.find(v => v.localService && v.lang.startsWith("en")) ||
    voices.find(v => v.name === "Google UK English Male") ||
    null;
  console.log("Selected voice:", cachedVoice?.name, "| local:", cachedVoice?.localService);
}
if (window.speechSynthesis) {
  loadPreferredVoice();
  window.speechSynthesis.addEventListener("voiceschanged", loadPreferredVoice);
  window.addEventListener("beforeunload", () => window.speechSynthesis.cancel());
}
function speakText(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const plain = text.replace(/[#*_`~\[\]()>|\\-]/g, "").replace(/\n+/g, " ").trim();
  if (!plain) return;

  // Split into sentence chunks to avoid Chrome's ~15s TTS stall bug
  const chunks = plain.match(/[^.!?]+[.!?]*/g)?.map(s => s.trim()).filter(Boolean) ?? [plain];
  let index = 0;

  // Switch to talking avatar immediately — don't rely on onstart which Chrome can miss
  isSpeaking = true;
  setAvatarState('talking');

  function speakNext() {
    if (index >= chunks.length) { isSpeaking = false; setAvatarState('idle'); return; }
    const utterance = new SpeechSynthesisUtterance(chunks[index++]);
    utterance.lang = "en-GB";
    utterance.rate = 1;
    utterance.pitch = 1;
    if (cachedVoice) utterance.voice = cachedVoice;
    utterance.onend = speakNext;
    utterance.onerror = () => { isSpeaking = false; setAvatarState('idle'); };
    window.speechSynthesis.speak(utterance);
  }
  // Give Chrome a tick to process the cancel() before queuing new speech
  setTimeout(speakNext, 10);
}

function appendMessage(role, text, fromVoice = false) {
  const message = document.createElement("article");
  message.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "me" ? "You" : "Avatar";
  message.appendChild(label);

  const content =
    role === "bot" ? renderAvatarMessage(text) : renderUserMessage(text);
  message.appendChild(content);

  if (role === "me" && fromVoice) {
    const editBtn = document.createElement("button");
    editBtn.className = "edit-btn";
    editBtn.textContent = "✏️ Edit";
    editBtn.title = "Transcription wrong? Edit and resubmit.";
    editBtn.addEventListener("click", () => {
      // store this message for removal on resubmit; bot sibling resolved at send time
      pendingEditElements = [message];
      textInput.value = text;
      textInput.focus();
      statusEl.textContent = "Edit your question and press Send or Enter.";
    });
    message.appendChild(editBtn);
  }

  if (role === "bot") {
    message.appendChild(createFeedbackRow(text));
  }

  chatLog.appendChild(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setRequestState(pending) {
  isSending = pending;
  sendBtn.disabled = pending;
}

// Thinking overlay — created once, toggled via class
const avatarThinkingEl = document.createElement("div");
avatarThinkingEl.id = "avatar-thinking";
avatarThinkingEl.innerHTML = `
  <div class="thought-trail">
    <div class="thought-trail-dot"></div>
    <div class="thought-trail-dot"></div>
    <div class="thought-trail-dot"></div>
  </div>
  <div class="thought-bubble">
    <div class="thinking-dots"><span></span><span></span><span></span></div>
  </div>`;
document.body.appendChild(avatarThinkingEl);

function showThinking() {
  // Project the avatar head from 3D space to screen coords, then offset to the right
  if (avatarHeadPosition) {
    const v = avatarHeadPosition.clone().project(camera);
    const headX = (v.x * 0.5 + 0.5) * window.innerWidth;
    const headY = (-v.y * 0.5 + 0.5) * window.innerHeight;
    avatarThinkingEl.style.left = `${headX + 8}px`;
    avatarThinkingEl.style.top  = `${headY}px`;
  } else {
    avatarThinkingEl.style.left = "60%";
    avatarThinkingEl.style.top  = "18%";
  }
  avatarThinkingEl.classList.add("visible");
}

function hideThinking() {
  avatarThinkingEl.classList.remove("visible");
}

async function handleSend(questionText, fromVoice = false) {
  const q = (questionText ?? textInput.value).trim();
  if (!q) {
    return;
  }
  if (isSending) {
    statusEl.textContent = "Please wait for the current answer.";
    return;
  }

  // remove original voice message + its bot response if user edited and resubmitted
  if (pendingEditElements) {
    const [userMsg] = pendingEditElements;
    const botMsg = userMsg?.nextElementSibling;
    botMsg?.remove();
    userMsg?.remove();
    pendingEditElements = null;
  }

  appendMessage("me", q, fromVoice);
  textInput.value = "";

  // check for canned responses first
  const canned = checkCannedResponse(q);
  if (canned) {
    appendMessage("bot", canned);
    speakText(canned);
    lastAnswer = canned;
    statusEl.textContent = "Answer received. Ask another question any time.";
    setRequestState(false);
    return;
  }

  setRequestState(true);
  statusEl.textContent = "Avatar is thinking...";
  showThinking();

  const queryToSend = buildQuery(q);
  lastQuestion = q;

  try {
    const response = await fetch(QUERY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryToSend }),
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
    lastAnswer = answer;
    hideThinking();
    appendMessage("bot", answer);
    speakText(answer);
    statusEl.textContent = "Answer received. Ask another question any time.";
  } catch (error) {
    console.error("Chat request failed:", error);
    hideThinking();
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
  recognition.interimResults = true;
  recognition.continuous = false;

  recognition.onstart = () => {
    isListening = true;
    statusEl.textContent = "Listening… ask your question.";
    micBtn.textContent = "🛑";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    textInput.value = transcript;
    statusEl.textContent = "Listening…";
  };

  recognition.onerror = (event) => {
    if (event.error === "no-speech") {
      statusEl.textContent = "No speech detected. Try again or type your question.";
    } else {
      statusEl.textContent = `Mic error: ${event.error}. You can still type.`;
    }
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = "🎤";
    const transcript = textInput.value.trim();
    if (transcript) {
      statusEl.textContent = `Heard: "${transcript}"`;
      handleSend(transcript, true);
    } else if (!statusEl.textContent.startsWith("Mic error") && !statusEl.textContent.startsWith("No speech")) {
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
  micBtn.disabled = true;
  micBtn.title = "Speech-to-text not supported in this browser.";
  statusEl.textContent =
    "Speech-to-text not supported here. Please type your question.";
}
