// ---------------------------------------------------------------------------
// Bootstrap: get API base URL
// ---------------------------------------------------------------------------

let API_BASE = "";

async function init() {
  if (window.electronAPI) {
    API_BASE = await window.electronAPI.getApiBase();
  } else {
    // Fallback for dev/browser testing
    const params = new URLSearchParams(window.location.search);
    API_BASE = `http://127.0.0.1:${params.get("port") || 7860}`;
  }
  await loadSettings();
  await refreshVoiceDropdowns();
  setupEventListeners();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  // TTS tab
  ttsPersonaInfo: null, // loaded persona metadata (speech_style, etc.)

  // Persona tab
  refAudioBlob: null,   // recorded or uploaded audio Blob
  refAudioChanged: false,

  // Settings
  settings: {},
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  return res;
}

async function apiJSON(path, options = {}) {
  const res = await apiFetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(msg, type = "") {
  const bar = document.getElementById("status-bar");
  const text = document.getElementById("status-text");
  bar.className = "status-bar" + (type ? ` ${type}` : "");
  bar.classList.remove("hidden");
  text.textContent = msg;
}

function clearStatus() {
  document.getElementById("status-bar").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${tab}`).classList.add("active");
    });
  });
}

// ---------------------------------------------------------------------------
// Settings / Model management
// ---------------------------------------------------------------------------

async function loadSettings() {
  const data = await apiJSON("/api/settings");
  state.settings = data;

  // TTS model dropdown
  const ttsEl = document.getElementById("model-tts-size");
  ttsEl.innerHTML = data.model_variants.map(
    (v) => `<option value="${v}">${v}</option>`
  ).join("");
  ttsEl.value = data.model_size;
  updateModelInfo("tts", data.model_size, data.model_variants);

  // ASR model dropdown
  const asrEl = document.getElementById("model-asr-size");
  asrEl.innerHTML = data.asr_models.map(
    (v) => `<option value="${v}">${v}</option>`
  ).join("");
  asrEl.value = data.asr_model_size;
  updateModelInfo("asr", data.asr_model_size, data.asr_models);

  // LLM (GGUF) model dropdown
  populateLlmDropdown(data.llm_models, data.llm_model);

  // Language dropdown in persona tab
  const langEl = document.getElementById("ref-language");
  langEl.innerHTML = data.languages.map(
    (l) => `<option value="${l}">${l}</option>`
  ).join("");
  langEl.value = "English";
}

function updateModelInfo(type, value, variants) {
  const el = document.getElementById(`model-${type}-info`);
  if (el) el.textContent = `Qwen/Qwen3-${type.toUpperCase()}-12Hz-${value}-Base`;
}

function populateLlmDropdown(models, selected) {
  const el = document.getElementById("model-llm-file");
  el.innerHTML = '<option value="">-- GGUFファイルを選択 --</option>';
  models.forEach((m) => {
    el.innerHTML += `<option value="${m}">${m}</option>`;
  });
  if (selected && models.includes(selected)) {
    el.value = selected;
    document.getElementById("model-llm-info").textContent = selected;
  }
}

async function saveModelSetting(key, value) {
  try {
    await apiJSON("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [key]: value }),
    });
    state.settings[key] = value;
  } catch (e) {
    setStatus(`設定保存エラー: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Voice dropdowns
// ---------------------------------------------------------------------------

async function refreshVoiceDropdowns() {
  const voices = await apiJSON("/api/voices");

  const ttsDd = document.getElementById("tts-speaker-dd");
  const prevTts = ttsDd.value;
  ttsDd.innerHTML = '<option value="">-- 選択 --</option>';
  voices.forEach((v) => {
    ttsDd.innerHTML += `<option value="${v}">${v}</option>`;
  });
  if (voices.includes(prevTts)) ttsDd.value = prevTts;

  const personaDd = document.getElementById("persona-dd");
  const prevPersona = personaDd.value;
  personaDd.innerHTML = '<option value="">-- 選択 --</option>';
  voices.forEach((v) => {
    personaDd.innerHTML += `<option value="${v}">${v}</option>`;
  });
  if (voices.includes(prevPersona)) personaDd.value = prevPersona;
}

// ---------------------------------------------------------------------------
// TTS tab
// ---------------------------------------------------------------------------

async function onTtsSpeakerChange() {
  const name = document.getElementById("tts-speaker-dd").value;
  document.getElementById("tts-speaker-label").textContent = name || "未選択";
  if (!name) {
    state.ttsPersonaInfo = null;
    return;
  }
  try {
    state.ttsPersonaInfo = await apiJSON(`/api/voices/${encodeURIComponent(name)}`);
  } catch (e) {
    state.ttsPersonaInfo = null;
  }
}

async function onGenerateLine() {
  const prompt = document.getElementById("tts-prompt").value.trim();
  if (!prompt) { setStatus("セリフ生成の指示を入力してください。", "error"); return; }

  const info = state.ttsPersonaInfo || {};
  const llmModel = state.settings.llm_model || "";

  setStatus("セリフ生成中...", "loading");
  document.getElementById("tts-gen-line-btn").disabled = true;
  const textEl = document.getElementById("tts-text");
  textEl.value = "";

  try {
    const response = await fetch(API_BASE + "/api/generate-line", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        language: info.language || "Japanese",
        speech_style: info.speech_style || "",
        phrase_bank: info.phrase_bank || "",
        speech_habits: info.speech_habits || "",
        ng_phrases: info.ng_phrases || "",
        sample_lines: info.sample_lines || "",
        llm_model: llmModel,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split("\n");
      sseBuffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") break;
        try {
          const parsed = JSON.parse(payload);
          if (parsed.token) textEl.value += parsed.token;
        } catch {}
      }
    }
    clearStatus();
  } catch (e) {
    setStatus(`エラー: ${e.message}`, "error");
  } finally {
    document.getElementById("tts-gen-line-btn").disabled = false;
  }
}

async function onGenerateTts() {
  const text = document.getElementById("tts-text").value.trim();
  if (!text) { setStatus("テキストを入力してください。", "error"); return; }

  const voiceName = document.getElementById("tts-speaker-dd").value;
  if (!voiceName) { setStatus("話者を選択してください。", "error"); return; }

  const modelSize = state.settings.model_size || "1.7B";

  setStatus("音声生成中...", "loading");
  document.getElementById("tts-gen-btn").disabled = true;
  const audioEl = document.getElementById("tts-audio");
  audioEl.src = "";

  try {
    const data = await apiJSON("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice_name: voiceName, model_size: modelSize }),
    });
    audioEl.src = API_BASE + data.audio_url + "?t=" + Date.now();
    audioEl.load();
    document.getElementById("tts-elapsed").textContent = data.elapsed;
    clearStatus();
  } catch (e) {
    setStatus(`エラー: ${e.message}`, "error");
  } finally {
    document.getElementById("tts-gen-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Persona tab – audio input (upload / mic)
// ---------------------------------------------------------------------------

const wavRecorder = (() => {
  let audioCtx = null;
  let processor = null;
  let source = null;
  let stream = null;
  let chunks = [];
  let sampleRate = 16000;

  function encodeWAV(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);
    const write = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    write(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, samples.length * 2, true);
    const int16 = new Int16Array(buf, 44);
    for (let i = 0; i < samples.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  return {
    async start() {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new AudioContext({ sampleRate });
      source = audioCtx.createMediaStreamSource(stream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      chunks = [];
      processor.onaudioprocess = (e) => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
    },
    stop() {
      source.disconnect();
      processor.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      const total = chunks.reduce((s, c) => s + c.length, 0);
      const combined = new Float32Array(total);
      let off = 0;
      for (const c of chunks) { combined.set(c, off); off += c.length; }
      return encodeWAV(combined, sampleRate);
    },
  };
})();

let isRecording = false;

async function onRecordToggle() {
  const btn = document.getElementById("record-btn");
  const statusEl = document.getElementById("record-status");

  if (!isRecording) {
    try {
      await wavRecorder.start();
      isRecording = true;
      btn.textContent = "録音停止";
      btn.classList.add("recording");
      statusEl.textContent = "録音中...";
    } catch (e) {
      setStatus("マイクへのアクセスが拒否されました: " + e.message, "error");
    }
  } else {
    const blob = wavRecorder.stop();
    isRecording = false;
    btn.textContent = "録音開始";
    btn.classList.remove("recording");
    statusEl.textContent = "録音完了";
    setRefAudioBlob(blob);
  }
}

function setRefAudioBlob(blob) {
  state.refAudioBlob = blob;
  state.refAudioChanged = true;
  const wrap = document.getElementById("ref-audio-preview-wrap");
  const preview = document.getElementById("ref-audio-preview");
  wrap.classList.remove("hidden");
  preview.src = URL.createObjectURL(blob);
  preview.load();
}

// ---------------------------------------------------------------------------
// Persona tab – CRUD
// ---------------------------------------------------------------------------

async function onPersonaLoad() {
  const name = document.getElementById("persona-dd").value;
  if (!name) { setStatus("声を選択してください。", "error"); return; }

  try {
    const info = await apiJSON(`/api/voices/${encodeURIComponent(name)}`);

    document.getElementById("persona-name-input").value = name;
    document.getElementById("ref-transcript").value = info.transcript || "";
    document.getElementById("ref-language").value = info.language || "Auto";
    document.getElementById("persona-speech-style").value = info.speech_style || "";
    document.getElementById("persona-phrase-bank").value = info.phrase_bank || "";
    document.getElementById("persona-speech-habits").value = info.speech_habits || "";
    document.getElementById("persona-ng-phrases").value = info.ng_phrases || "";
    document.getElementById("persona-sample-lines").value = info.sample_lines || "";

    // Load ref audio for preview
    const audioUrl = `${API_BASE}/api/voices/${encodeURIComponent(name)}/audio`;
    const wrap = document.getElementById("ref-audio-preview-wrap");
    const preview = document.getElementById("ref-audio-preview");
    wrap.classList.remove("hidden");
    preview.src = audioUrl + "?t=" + Date.now();
    preview.load();

    // Mark audio as unchanged (existing file on disk)
    state.refAudioBlob = null;
    state.refAudioChanged = false;
    document.getElementById("ref-audio-file").value = "";

    clearStatus();
  } catch (e) {
    setStatus(`読み込みエラー: ${e.message}`, "error");
  }
}

async function onPersonaDelete() {
  const name = document.getElementById("persona-dd").value;
  if (!name) { setStatus("削除する声を選択してください。", "error"); return; }
  if (!confirm(`「${name}」を削除しますか？`)) return;

  try {
    const data = await apiJSON(`/api/voices/${encodeURIComponent(name)}`, { method: "DELETE" });
    await refreshVoiceDropdowns();
    clearStatus();
  } catch (e) {
    setStatus(`削除エラー: ${e.message}`, "error");
  }
}

async function savePersona(nameOverride) {
  const name = (nameOverride || document.getElementById("persona-name-input").value).trim();
  if (!name) { setStatus("登録名を入力してください。", "error"); return; }

  const formData = new FormData();
  formData.append("name", name);
  formData.append("transcript", document.getElementById("ref-transcript").value);
  formData.append("language", document.getElementById("ref-language").value);
  formData.append("speech_style", document.getElementById("persona-speech-style").value);
  formData.append("phrase_bank", document.getElementById("persona-phrase-bank").value);
  formData.append("speech_habits", document.getElementById("persona-speech-habits").value);
  formData.append("ng_phrases", document.getElementById("persona-ng-phrases").value);
  formData.append("sample_lines", document.getElementById("persona-sample-lines").value);

  // Audio: prefer recorded blob, then file upload
  const fileInput = document.getElementById("ref-audio-file");
  if (state.refAudioBlob && state.refAudioChanged) {
    formData.append("audio", state.refAudioBlob, "ref.wav");
  } else if (fileInput.files.length > 0) {
    formData.append("audio", fileInput.files[0]);
  }

  setStatus("保存中...", "loading");
  try {
    const data = await apiJSON("/api/voices", { method: "POST", body: formData });
    await refreshVoiceDropdowns();
    document.getElementById("persona-dd").value = name;
    clearStatus();
    setStatus(`「${name}」を保存しました。`);
    setTimeout(clearStatus, 2000);
  } catch (e) {
    setStatus(`保存エラー: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Persona tab – Transcribe
// ---------------------------------------------------------------------------

async function onTranscribe() {
  const fileInput = document.getElementById("ref-audio-file");
  const hasFile = fileInput.files.length > 0;
  const hasBlob = state.refAudioBlob !== null;

  if (!hasFile && !hasBlob) {
    setStatus("参照音声をアップロードまたは録音してください。", "error");
    return;
  }

  const formData = new FormData();
  if (state.refAudioBlob && state.refAudioChanged) {
    formData.append("audio", state.refAudioBlob, "ref.wav");
  } else {
    formData.append("audio", fileInput.files[0]);
  }
  formData.append("asr_size", state.settings.asr_model_size || "0.6B");

  setStatus("文字起こし中...", "loading");
  document.getElementById("transcribe-btn").disabled = true;
  try {
    const data = await apiJSON("/api/transcribe", { method: "POST", body: formData });
    document.getElementById("ref-transcript").value = data.text;
    document.getElementById("ref-language").value = data.language;
    clearStatus();
  } catch (e) {
    setStatus(`文字起こしエラー: ${e.message}`, "error");
  } finally {
    document.getElementById("transcribe-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Audio source toggle (upload / mic)
// ---------------------------------------------------------------------------

function setupAudioSourceToggle() {
  // File input change: preview audio
  document.getElementById("ref-audio-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    state.refAudioBlob = null;
    state.refAudioChanged = false;
    const wrap = document.getElementById("ref-audio-preview-wrap");
    const preview = document.getElementById("ref-audio-preview");
    wrap.classList.remove("hidden");
    preview.src = URL.createObjectURL(file);
    preview.load();
  });
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

function setupModelControls() {
  document.getElementById("model-tts-size").addEventListener("change", async (e) => {
    const v = e.target.value;
    document.getElementById("model-tts-info").textContent = `Qwen/Qwen3-TTS-12Hz-${v}-Base`;
    await saveModelSetting("model_size", v);
  });

  document.getElementById("model-asr-size").addEventListener("change", async (e) => {
    const v = e.target.value;
    document.getElementById("model-asr-info").textContent = `Qwen/Qwen3-ASR-${v}`;
    await saveModelSetting("asr_model_size", v);
  });

  document.getElementById("model-llm-file").addEventListener("change", async (e) => {
    const v = e.target.value;
    document.getElementById("model-llm-info").textContent = v || "";
    state.settings.llm_model = v;
    await saveModelSetting("llm_model", v);
  });

  document.getElementById("model-llm-refresh-btn").addEventListener("click", async () => {
    try {
      const models = await apiJSON("/api/llm-models");
      populateLlmDropdown(models, state.settings.llm_model || "");
    } catch (e) {
      setStatus(`更新エラー: ${e.message}`, "error");
    }
  });
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

function setupEventListeners() {
  setupTabs();
  setupAudioSourceToggle();
  setupModelControls();

  // TTS tab
  document.getElementById("tts-speaker-dd").addEventListener("change", onTtsSpeakerChange);
  document.getElementById("tts-gen-line-btn").addEventListener("click", onGenerateLine);
  document.getElementById("tts-gen-btn").addEventListener("click", onGenerateTts);

  // Persona tab
  document.getElementById("persona-load-btn").addEventListener("click", onPersonaLoad);
  document.getElementById("persona-delete-btn").addEventListener("click", onPersonaDelete);
  document.getElementById("persona-save-btn").addEventListener("click", () => savePersona(null));
  document.getElementById("persona-overwrite-btn").addEventListener("click", () => {
    const selected = document.getElementById("persona-dd").value;
    savePersona(selected || null);
  });
  document.getElementById("transcribe-btn").addEventListener("click", onTranscribe);
  document.getElementById("record-btn").addEventListener("click", onRecordToggle);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

init().catch((e) => {
  console.error("Init failed:", e);
  document.body.innerHTML = `<div style="padding:20px;color:#e94560">
    サーバーへの接続に失敗しました。<br>${e.message}
  </div>`;
});
