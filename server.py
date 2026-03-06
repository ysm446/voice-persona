import os

# HF_HOME must be set before any HuggingFace/transformers imports
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from flask import Flask, request, jsonify, send_file, abort, Response, stream_with_context
from flask_cors import CORS
from qwen_tts import Qwen3TTSModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MODEL_VARIANTS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

ASR_MODELS = {
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
}

ASR_LANGUAGE_MAP = {
    "Chinese": "Chinese",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "German": "German",
    "French": "French",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Italian": "Italian",
}

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

PERSONAS_DIR = Path(__file__).parent / "persona"
LEGACY_VOICES_DIR = Path(__file__).parent / "voices"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
LLM_DIR = Path(__file__).parent / "models" / "gguf"
SETTINGS_PATH = Path(__file__).parent / "settings.json"
OUTPUTS_DIR.mkdir(exist_ok=True)
LLM_DIR.mkdir(exist_ok=True)

_models: dict = {}

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(data: dict):
    existing = _load_settings()
    existing.update(data)
    SETTINGS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

def _ensure_persona_dir():
    PERSONAS_DIR.mkdir(exist_ok=True)
    if not LEGACY_VOICES_DIR.exists():
        return
    import shutil
    for legacy_dir in LEGACY_VOICES_DIR.iterdir():
        if not legacy_dir.is_dir():
            continue
        target_dir = PERSONAS_DIR / legacy_dir.name
        if target_dir.exists():
            continue
        shutil.move(str(legacy_dir), str(target_dir))


def _list_voices() -> list[str]:
    _ensure_persona_dir()
    return sorted(d.name for d in PERSONAS_DIR.iterdir() if d.is_dir())


def _yaml_dump(data: dict) -> str:
    class _BlockStyleDumper(yaml.SafeDumper):
        pass

    def _represent_str(dumper, value: str):
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _BlockStyleDumper.add_representer(str, _represent_str)
    return yaml.dump(
        data,
        Dumper=_BlockStyleDumper,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    )


def _write_persona_info(voice_dir: Path, info: dict) -> None:
    (voice_dir / "info.yaml").write_text(_yaml_dump(info), encoding="utf-8")


def _read_persona_info(voice_dir: Path) -> dict:
    yaml_path = voice_dir / "info.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    legacy_path = voice_dir / "info.json"
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Persona info not found: {voice_dir}")


def _read_audio_upload(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """Read uploaded audio bytes, return (data_float32, sample_rate)."""
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        return data.astype(np.float32), sr
    except Exception:
        pass
    # Fallback via torchaudio (handles WebM/Opus from browser MediaRecorder)
    import torchaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
        data = waveform.mean(0).numpy().astype(np.float32)
        return data, sr
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# MP3 export
# ---------------------------------------------------------------------------

def _sanitize_filename_part(value: str | None) -> str:
    if not value:
        return "unknown"
    sanitized = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def _to_mp3(data: np.ndarray, sr: int, speaker_name: str | None = None) -> str:
    import lameenc
    if data.dtype.kind == "f":
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_bytes = encoder.encode(data.tobytes()) + encoder.flush()
    speaker = _sanitize_filename_part(speaker_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUTS_DIR / f"{speaker}_{timestamp}.mp3"
    output_path.write_bytes(mp3_bytes)
    return str(output_path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_tts_model(size: str) -> Qwen3TTSModel:
    size = size if size in MODEL_VARIANTS else "1.7B"
    cache_key = f"tts:{size}"
    if cache_key not in _models:
        model_id = MODEL_VARIANTS[size]
        print(f"[voice-persona] Loading {model_id} ...")
        load_kwargs: dict = {"device_map": DEVICE, "dtype": DTYPE}
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[voice-persona] flash_attention_2 enabled")
        except ImportError:
            pass
        _models[cache_key] = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        print(f"[voice-persona] Loaded: {model_id}")
    return _models[cache_key]


def get_asr_model(size: str):
    size = size if size in ASR_MODELS else "0.6B"
    cache_key = f"asr:{size}"
    if cache_key not in _models:
        from qwen_asr import Qwen3ASRModel
        model_id = ASR_MODELS[size]
        print(f"[voice-persona] Loading {model_id} ...")
        _models[cache_key] = Qwen3ASRModel.from_pretrained(
            model_id, dtype=DTYPE, device_map=DEVICE,
        )
        print(f"[voice-persona] Loaded: {model_id}")
    return _models[cache_key]


def _list_gguf_models() -> list[str]:
    return sorted(f.name for f in LLM_DIR.glob("*.gguf"))


def get_llm_model(filename: str):
    if not filename:
        raise ValueError("LLMモデルが選択されていません。Model管理タブでGGUFファイルを選択してください。")
    cache_key = f"llm:{filename}"
    if cache_key not in _models:
        from llama_cpp import Llama
        model_path = LLM_DIR / filename
        if not model_path.exists():
            raise ValueError(f"モデルファイルが見つかりません: {filename}")
        print(f"[voice-persona] Loading {filename} ...")
        _models[cache_key] = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        print(f"[voice-persona] Loaded: {filename}")
    return _models[cache_key]


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def _generate_tts(text: str, voice_name: str, model_size: str) -> tuple[str, str]:
    voice_dir = PERSONAS_DIR / voice_name
    if not voice_dir.exists():
        raise ValueError(f"Persona not found: {voice_name}")

    ref_data, ref_sr = sf.read(str(voice_dir / "ref.wav"), always_2d=False)
    info = _read_persona_info(voice_dir)
    language = info.get("language", "Auto")
    ref_text = info.get("transcript", "")

    if not ref_text.strip():
        raise ValueError("参照音声のトランスクリプトがありません。Personaタブで設定してください。")

    ref_data = ref_data.astype(np.float32)
    max_val = np.abs(ref_data).max()
    if max_val > 1.0:
        ref_data = ref_data / max_val

    model = get_tts_model(model_size)
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=(ref_data, ref_sr),
        ref_text=ref_text,
    )
    elapsed = time.perf_counter() - t0
    mp3_path = _to_mp3(wavs[0], sr, speaker_name=voice_name)
    return Path(mp3_path).name, f"{elapsed:.1f} 秒"


def _build_generate_line_args(
    prompt: str,
    language: str,
    speech_style: str,
    phrase_bank: str,
    speech_habits: str,
    ng_phrases: str,
    sample_lines: str,
) -> list[dict]:
    persona_sheet = (
        f"話し方プロファイル:\n{speech_style.strip() or '(未設定)'}\n\n"
        f"言い回し集:\n{phrase_bank.strip() or '(未設定)'}\n\n"
        f"口癖:\n{speech_habits.strip() or '(未設定)'}\n\n"
        f"NG表現:\n{ng_phrases.strip() or '(未設定)'}\n\n"
        f"サンプル台詞:\n{sample_lines.strip() or '(未設定)'}"
    )
    instruction = (
        "以下のPersona設定に沿って、指定シーンで人物が話しそうなセリフを1つ作成してください。"
        f"出力言語は {language}。"
        "地の文や説明は不要で、セリフ本文のみを返してください。"
    )
    user_content = f"{instruction}\n\n[Persona設定]\n{persona_sheet}\n\n[依頼]\n{prompt.strip()}"
    return [
        {"role": "system", "content": "You are a dialogue writer for TTS scripts."},
        {"role": "user", "content": user_content},
    ]


def _stream_tokens(llm, messages: list[dict]):
    """SSE generator that streams tokens, filtering <think>...</think> blocks."""
    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"
    in_think = False
    buf = ""

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        stream=True,
    ):
        delta = (chunk["choices"][0]["delta"].get("content") or "")
        if not delta:
            continue
        buf += delta
        output = ""

        while buf:
            if in_think:
                idx = buf.find(CLOSE_TAG)
                if idx >= 0:
                    buf = buf[idx + len(CLOSE_TAG):]
                    in_think = False
                else:
                    # Keep potential partial closing tag in buffer
                    keep = next(
                        (i for i in range(len(CLOSE_TAG) - 1, 0, -1) if buf.endswith(CLOSE_TAG[:i])),
                        0,
                    )
                    buf = buf[-keep:] if keep else ""
                    break
            else:
                idx = buf.find(OPEN_TAG)
                if idx >= 0:
                    output += buf[:idx]
                    buf = buf[idx + len(OPEN_TAG):]
                    in_think = True
                else:
                    # Keep potential partial opening tag in buffer
                    keep = next(
                        (i for i in range(len(OPEN_TAG) - 1, 0, -1) if buf.endswith(OPEN_TAG[:i])),
                        0,
                    )
                    if keep:
                        output += buf[:-keep]
                        buf = buf[-keep:]
                    else:
                        output += buf
                        buf = ""
                    break

        if output:
            yield f"data: {json.dumps({'token': output}, ensure_ascii=False)}\n\n"

    if buf and not in_think:
        yield f"data: {json.dumps({'token': buf}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/voices")
def list_voices():
    return jsonify(_list_voices())


@app.route("/api/voices/<name>")
def get_voice(name):
    voice_dir = PERSONAS_DIR / name
    if not voice_dir.exists():
        return abort(404)
    info = _read_persona_info(voice_dir)
    return jsonify(info)


@app.route("/api/voices/<name>/audio")
def get_voice_audio(name):
    audio_path = PERSONAS_DIR / name / "ref.wav"
    if not audio_path.exists():
        return abort(404)
    return send_file(str(audio_path), mimetype="audio/wav")


@app.route("/api/voices", methods=["POST"])
def save_voice():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "登録名を入力してください。"}), 400

    _ensure_persona_dir()
    voice_dir = PERSONAS_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)

    audio_file = request.files.get("audio")
    if audio_file:
        audio_bytes = audio_file.read()
        data, sr = _read_audio_upload(audio_bytes)
        sf.write(str(voice_dir / "ref.wav"), data, sr)
    elif not (voice_dir / "ref.wav").exists():
        return jsonify({"error": "参照音声をアップロードしてください。"}), 400

    _write_persona_info(voice_dir, {
        "transcript": request.form.get("transcript", ""),
        "language": request.form.get("language", "Auto"),
        "speech_style": request.form.get("speech_style", ""),
        "phrase_bank": request.form.get("phrase_bank", ""),
        "speech_habits": request.form.get("speech_habits", ""),
        "ng_phrases": request.form.get("ng_phrases", ""),
        "sample_lines": request.form.get("sample_lines", ""),
    })
    return jsonify({"success": True, "voices": _list_voices()})


@app.route("/api/voices/<name>", methods=["DELETE"])
def delete_voice(name):
    import shutil
    voice_dir = PERSONAS_DIR / name
    if voice_dir.exists():
        shutil.rmtree(str(voice_dir))
    return jsonify({"success": True, "voices": _list_voices()})


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.json
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "テキストを入力してください。"}), 400
    voice_name = data.get("voice_name")
    if not voice_name:
        return jsonify({"error": "話者を選択してください。"}), 400
    settings = _load_settings()
    model_size = data.get("model_size") or settings.get("model_size", "1.7B")
    try:
        filename, elapsed = _generate_tts(text, voice_name, model_size)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"audio_url": f"/outputs/{filename}", "elapsed": elapsed})


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    settings = _load_settings()
    asr_size = request.form.get("asr_size") or settings.get("asr_model_size", "0.6B")
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "参照音声をアップロードしてください。"}), 400

    audio_bytes = audio_file.read()
    data, sr = _read_audio_upload(audio_bytes)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, data, sr)
        model = get_asr_model(asr_size)
        results = model.transcribe(audio=tmp_path, language=None)
    finally:
        os.unlink(tmp_path)

    result = results[0]
    text = result.text
    detected = getattr(result, "language", None) or ""
    language = ASR_LANGUAGE_MAP.get(detected, "Auto")
    return jsonify({"text": text, "language": language})


@app.route("/api/generate-line", methods=["POST"])
def generate_line():
    data = request.json
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "セリフ生成の指示を入力してください。"}), 400
    settings = _load_settings()
    llm_model = data.get("llm_model") or settings.get("llm_model", "")
    try:
        llm = get_llm_model(llm_model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    messages = _build_generate_line_args(
        prompt=prompt,
        language=data.get("language", "Japanese"),
        speech_style=data.get("speech_style", ""),
        phrase_bank=data.get("phrase_bank", ""),
        speech_habits=data.get("speech_habits", ""),
        ng_phrases=data.get("ng_phrases", ""),
        sample_lines=data.get("sample_lines", ""),
    )

    def sse():
        yield from _stream_tokens(llm, messages)

    return Response(
        stream_with_context(sse()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/settings")
def get_settings():
    settings = _load_settings()
    return jsonify({
        "model_size": settings.get("model_size", "1.7B"),
        "asr_model_size": settings.get("asr_model_size", "0.6B"),
        "llm_model": settings.get("llm_model", ""),
        "model_variants": list(MODEL_VARIANTS.keys()),
        "asr_models": list(ASR_MODELS.keys()),
        "llm_models": _list_gguf_models(),
        "languages": LANGUAGES,
    })


@app.route("/api/llm-models")
def list_llm_models():
    return jsonify(_list_gguf_models())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    allowed = {"model_size", "asr_model_size", "llm_model"}
    filtered = {k: v for k, v in data.items() if k in allowed}
    _save_settings(filtered)
    return jsonify({"success": True})


@app.route("/outputs/<filename>")
def serve_output(filename):
    filepath = OUTPUTS_DIR / filename
    if not filepath.exists():
        return abort(404)
    return send_file(str(filepath), mimetype="audio/mpeg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket

    env_port = os.environ.get("APP_PORT", "")
    if env_port.isdigit():
        port = int(env_port)
    else:
        port = 7860
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    break
            port += 1

    print(f"[voice-persona] Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
