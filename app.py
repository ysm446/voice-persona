import os

# HF_HOME must be set before any HuggingFace/transformers imports
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import gradio as gr
import yaml
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

WRITER_MODELS = {
    "1.7B": "Qwen/Qwen3-1.7B",
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "Huihui-8B-v2": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
    "14B": "Qwen/Qwen3-14B",
}

WRITER_MODEL_CHOICES = [
    ("0.6B", "0.6B"),
    ("1.7B", "1.7B"),
    ("4B", "4B"),
    ("8B", "8B"),
    ("Huihui 8B v2", "Huihui-8B-v2"),
    ("14B", "14B"),
]

# Map ASR-detected language names to TTS language names
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
SETTINGS_PATH = Path(__file__).parent / "settings.json"
OUTPUTS_DIR.mkdir(exist_ok=True)

_models: dict = {}


# ---------------------------------------------------------------------------
# Registered voice helpers
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


def _load_model_size_setting() -> str:
    size = _load_settings().get("model_size", "1.7B")
    return size if size in MODEL_VARIANTS else "1.7B"


def _save_model_size_setting(size: str):
    if size not in MODEL_VARIANTS:
        return
    _save_settings({"model_size": size})


def _load_asr_model_size_setting() -> str:
    size = _load_settings().get("asr_model_size", "0.6B")
    return size if size in ASR_MODELS else "0.6B"


def _save_asr_model_size_setting(size: str):
    if size not in ASR_MODELS:
        return
    _save_settings({"asr_model_size": size})


def _load_writer_model_size_setting() -> str:
    size = _load_settings().get("writer_model_size", "0.6B")
    return size if size in WRITER_MODELS else "0.6B"


def _save_writer_model_size_setting(size: str):
    if size not in WRITER_MODELS:
        return
    _save_settings({"writer_model_size": size})


def update_model_size(size: str) -> str:
    size = size if size in MODEL_VARIANTS else "1.7B"
    _save_model_size_setting(size)
    return render_model_info(size)


def update_asr_model_size(size: str) -> str:
    size = size if size in ASR_MODELS else "0.6B"
    _save_asr_model_size_setting(size)
    return f"`{ASR_MODELS[size]}`"


def update_writer_model_size(size: str) -> str:
    size = size if size in WRITER_MODELS else "0.6B"
    _save_writer_model_size_setting(size)
    return f"`{WRITER_MODELS[size]}`"


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


def _speaker_label(name: str | None) -> str:
    return name if name else "未選択"


def _persona_info_path(voice_dir: Path) -> Path:
    return voice_dir / "info.yaml"


def _legacy_persona_info_path(voice_dir: Path) -> Path:
    return voice_dir / "info.json"


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
    _persona_info_path(voice_dir).write_text(_yaml_dump(info), encoding="utf-8")


def _read_persona_info(voice_dir: Path) -> dict:
    yaml_path = _persona_info_path(voice_dir)
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    legacy_path = _legacy_persona_info_path(voice_dir)
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Persona info not found: {voice_dir}")


def save_voice(
    name: str,
    ref_audio,
    ref_text: str,
    language: str,
    speech_style: str,
    phrase_bank: str,
    speech_habits: str,
    ng_phrases: str,
    sample_lines: str,
):
    name = name.strip()
    if not name:
        raise gr.Error("登録名を入力してください。")
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    _ensure_persona_dir()
    voice_dir = PERSONAS_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    ref_sr, ref_data = ref_audio
    sf.write(str(voice_dir / "ref.wav"), ref_data, ref_sr)
    _write_persona_info(
        voice_dir,
        {
            "transcript": ref_text,
            "language": language,
            "speech_style": speech_style,
            "phrase_bank": phrase_bank,
            "speech_habits": speech_habits,
            "ng_phrases": ng_phrases,
            "sample_lines": sample_lines,
        },
    )
    return gr.Dropdown(choices=_list_voices(), value=name), _speaker_label(name)


def delete_voice(name: str):
    if not name:
        raise gr.Error("削除する声を選択してください。")
    import shutil
    _ensure_persona_dir()
    voice_dir = PERSONAS_DIR / name
    if voice_dir.exists():
        shutil.rmtree(str(voice_dir))
    voices = _list_voices()
    selected = voices[0] if voices else None
    return gr.Dropdown(choices=voices, value=selected), _speaker_label(selected)


def load_voice(name: str):
    if not name:
        raise gr.Error("声を選択してください。")
    _ensure_persona_dir()
    voice_dir = PERSONAS_DIR / name
    data, sr = sf.read(str(voice_dir / "ref.wav"), always_2d=False)
    info = _read_persona_info(voice_dir)
    return (
        (sr, data),
        info.get("transcript", ""),
        info.get("language", "Auto"),
        info.get("speech_style", ""),
        info.get("phrase_bank", ""),
        info.get("speech_habits", ""),
        info.get("ng_phrases", ""),
        info.get("sample_lines", ""),
    )


def load_voice_for_form(name: str):
    return (*load_voice(name), name, _speaker_label(name))


def overwrite_voice(
    selected_name: str,
    ref_audio,
    ref_text: str,
    language: str,
    speech_style: str,
    phrase_bank: str,
    speech_habits: str,
    ng_phrases: str,
    sample_lines: str,
):
    if not selected_name:
        raise gr.Error("上書きする声を選択してください。")
    return save_voice(
        selected_name,
        ref_audio,
        ref_text,
        language,
        speech_style,
        phrase_bank,
        speech_habits,
        ng_phrases,
        sample_lines,
    )


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
    # Convert float32 [-1, 1] → int16
    if data.dtype.kind == "f":
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    # Downmix to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = highest quality
    mp3_bytes = encoder.encode(data.tobytes()) + encoder.flush()
    speaker = _sanitize_filename_part(speaker_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUTS_DIR / f"{speaker}_{timestamp}.mp3"
    output_path.write_bytes(mp3_bytes)
    return str(output_path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_asr_model(size: str):
    size = size if size in ASR_MODELS else "0.6B"
    cache_key = f"asr:{size}"
    if cache_key not in _models:
        from qwen_asr import Qwen3ASRModel
        model_id = ASR_MODELS[size]
        print(f"[voice-echo] Loading {model_id} ...")
        _models[cache_key] = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE,
        )
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def get_tts_model(size: str) -> Qwen3TTSModel:
    size = size if size in MODEL_VARIANTS else "1.7B"
    model_id = MODEL_VARIANTS[size]
    cache_key = f"{size}:voice_clone"
    if cache_key not in _models:
        print(f"[voice-echo] Loading {model_id} ...")
        load_kwargs: dict = {"device_map": DEVICE, "dtype": DTYPE}
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[voice-echo] flash_attention_2 enabled")
        except ImportError:
            pass
        _models[cache_key] = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def render_model_info(size: str) -> str:
    size = size if size in MODEL_VARIANTS else "1.7B"
    return f"- Voice Clone: `{MODEL_VARIANTS[size]}`"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_clone(
    text: str,
    language: str,
    ref_audio,
    ref_text: str,
    model_size: str,
    speaker_name: str,
):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    if not ref_text.strip():
        raise gr.Error("参照音声のトランスクリプトを入力してください。")
    model = get_tts_model(model_size)
    # Gradio delivers (sample_rate, numpy_array); qwen_tts expects (numpy_array, sample_rate)
    ref_sr, ref_data = ref_audio
    # Normalize to float32 [-1, 1]; Gradio may return raw PCM values (e.g. int16 range as float)
    ref_data = ref_data.astype(np.float32)
    max_val = np.abs(ref_data).max()
    if max_val > 1.0:
        ref_data = ref_data / max_val
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=(ref_data, ref_sr),
        ref_text=ref_text,
    )
    elapsed = time.perf_counter() - t0
    return _to_mp3(wavs[0], sr, speaker_name=speaker_name), f"{elapsed:.1f} 秒"


def transcribe_ref(ref_audio, asr_size: str):
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    model = get_asr_model(asr_size)
    ref_sr, ref_data = ref_audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, ref_data, ref_sr)
        results = model.transcribe(audio=tmp_path, language=None)
    finally:
        os.unlink(tmp_path)
    result = results[0]
    text = result.text
    detected = getattr(result, "language", None) or ""
    language = ASR_LANGUAGE_MAP.get(detected, "Auto")
    return text, language


def get_writer_model(size: str):
    size = size if size in WRITER_MODELS else "0.6B"
    cache_key = f"writer:{size}"
    if cache_key not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = WRITER_MODELS[size]
        print(f"[voice-echo] Loading {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        _models[cache_key] = (model, tokenizer)
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def generate_persona_line(
    prompt: str,
    language: str,
    speech_style: str,
    phrase_bank: str,
    speech_habits: str,
    ng_phrases: str,
    sample_lines: str,
    writer_size: str,
):
    if not prompt.strip():
        raise gr.Error("セリフ生成の指示を入力してください。")
    model, tokenizer = get_writer_model(writer_size)
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
    messages = [
        {"role": "system", "content": "You are a dialogue writer for TTS scripts."},
        {"role": "user", "content": user_content},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            do_sample=True,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Voice Persona") as demo:
    default_model_size = _load_model_size_setting()
    default_asr_model_size = _load_asr_model_size_setting()
    default_writer_model_size = _load_writer_model_size_setting()
    gr.Markdown("# Voice Persona")
    gr.Markdown("人物設定から始める、自然な音声生成")

    with gr.Tabs():
        # --- Tab 1: Text-to-Speech ---
        with gr.Tab("テキスト読み上げ"):
            gr.Markdown(
                "Personaタブで作成した人物情報を使って、テキストを読み上げます。"
            )
            with gr.Row():
                with gr.Column():
                    current_speaker = gr.Textbox(
                        label="現在の話者",
                        value="未選択",
                        interactive=False,
                    )
                    vc_prompt = gr.Textbox(
                        label="セリフ生成の指示（Qwen）",
                        lines=2,
                        placeholder="例: 予算委員会で防衛政策について質問された時に答えそうなセリフ",
                    )
                    vc_generate_line_btn = gr.Button("🪄 セリフ生成", variant="secondary")
                    vc_text = gr.Textbox(
                        label="読み上げるテキスト",
                        lines=4,
                        placeholder="クローンした声で読み上げるテキストを入力...",
                    )
                    vc_btn = gr.Button("音声を生成", variant="primary")

                with gr.Column():
                    vc_audio = gr.Audio(label="出力音声")
                    vc_time = gr.Textbox(label="生成時間", interactive=False)

        with gr.Tab("人物作成・管理"):
            gr.Markdown("人物（参照音声とテキスト）を作成・管理します。")
            with gr.Group():
                gr.Markdown("#### 登録済みの声")
                with gr.Row():
                    persona_voice_dd = gr.Dropdown(
                        choices=_list_voices(),
                        label="声を選択",
                        scale=3,
                    )
                    persona_load_btn = gr.Button("読み込み", scale=1)
                    persona_delete_btn = gr.Button("🗑️ 削除", scale=1, variant="stop")
            with gr.Group():
                gr.Markdown("#### 現在の参照音声を登録")
                with gr.Row():
                    persona_voice_name = gr.Textbox(
                        label="登録名",
                        placeholder="例: MyVoice",
                        scale=3,
                    )
                    persona_save_btn = gr.Button("新規登録", scale=1)
                    persona_overwrite_btn = gr.Button("上書き保存", scale=1, variant="secondary")
            with gr.Group():
                gr.Markdown("#### 人物作成")
                vc_ref_audio = gr.Audio(
                    label="参照音声",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                vc_transcribe_btn = gr.Button("📝 自動文字起こし (ASR)")
                vc_ref_text = gr.Textbox(
                    label="参照音声のトランスクリプト",
                    lines=2,
                    placeholder="参照音声で話されている内容を正確に入力...",
                )
                vc_language = gr.Dropdown(
                    choices=LANGUAGES, value="English", label="言語"
                )
                persona_speech_style = gr.Textbox(
                    label="話し方プロファイル",
                    lines=3,
                    placeholder="丁寧さ、テンポ、語尾の傾向など",
                )
                persona_phrase_bank = gr.Textbox(
                    label="言い回し集",
                    lines=3,
                    placeholder="よく使う表現を改行区切りで入力",
                )
                persona_speech_habits = gr.Textbox(
                    label="口癖",
                    lines=2,
                    placeholder="繰り返しがちな語句や言い回し",
                )
                persona_ng_phrases = gr.Textbox(
                    label="NG表現",
                    lines=2,
                    placeholder="このPersonaで使わせたくない表現",
                )
                persona_sample_lines = gr.Textbox(
                    label="サンプル台詞",
                    lines=4,
                    placeholder="この人物らしいサンプル台詞を複数入力",
                )

        with gr.Tab("Model管理"):
            gr.Markdown("生成に使うモデルサイズを選択します。")
            gr.Markdown("### Voice Clone モデル")
            model_size = gr.Dropdown(
                choices=list(MODEL_VARIANTS.keys()),
                value=default_model_size,
                label="TTS モデルサイズ",
            )
            model_info = gr.Markdown(render_model_info(default_model_size))
            model_size.change(fn=update_model_size, inputs=model_size, outputs=model_info)
            gr.Markdown("### ASR モデル（自動文字起こし）")
            asr_model_size = gr.Dropdown(
                choices=list(ASR_MODELS.keys()),
                value=default_asr_model_size,
                label="ASR モデルサイズ",
            )
            asr_model_info = gr.Markdown(f"`{ASR_MODELS[default_asr_model_size]}`")
            asr_model_size.change(fn=update_asr_model_size, inputs=asr_model_size, outputs=asr_model_info)
            gr.Markdown("### セリフ生成モデル（Qwen）")
            writer_model_size = gr.Dropdown(
                choices=WRITER_MODEL_CHOICES,
                value=default_writer_model_size,
                label="セリフ生成モデルサイズ",
            )
            writer_model_info = gr.Markdown(f"`{WRITER_MODELS[default_writer_model_size]}`")
            writer_model_size.change(
                fn=update_writer_model_size,
                inputs=writer_model_size,
                outputs=writer_model_info,
            )

    vc_btn.click(
        fn=generate_clone,
        inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text, model_size, current_speaker],
        outputs=[vc_audio, vc_time],
    )
    vc_transcribe_btn.click(
        fn=transcribe_ref,
        inputs=[vc_ref_audio, asr_model_size],
        outputs=[vc_ref_text, vc_language],
    )
    vc_generate_line_btn.click(
        fn=generate_persona_line,
        inputs=[
            vc_prompt,
            vc_language,
            persona_speech_style,
            persona_phrase_bank,
            persona_speech_habits,
            persona_ng_phrases,
            persona_sample_lines,
            writer_model_size,
        ],
        outputs=[vc_text],
    )
    persona_load_btn.click(
        fn=load_voice_for_form,
        inputs=persona_voice_dd,
        outputs=[
            vc_ref_audio,
            vc_ref_text,
            vc_language,
            persona_speech_style,
            persona_phrase_bank,
            persona_speech_habits,
            persona_ng_phrases,
            persona_sample_lines,
            persona_voice_name,
            current_speaker,
        ],
    )
    persona_voice_dd.change(
        fn=_speaker_label,
        inputs=persona_voice_dd,
        outputs=current_speaker,
    )
    persona_delete_btn.click(
        fn=delete_voice,
        inputs=persona_voice_dd,
        outputs=[persona_voice_dd, current_speaker],
    )
    persona_save_btn.click(
        fn=save_voice,
        inputs=[
            persona_voice_name,
            vc_ref_audio,
            vc_ref_text,
            vc_language,
            persona_speech_style,
            persona_phrase_bank,
            persona_speech_habits,
            persona_ng_phrases,
            persona_sample_lines,
        ],
        outputs=[persona_voice_dd, current_speaker],
    )
    persona_overwrite_btn.click(
        fn=overwrite_voice,
        inputs=[
            persona_voice_dd,
            vc_ref_audio,
            vc_ref_text,
            vc_language,
            persona_speech_style,
            persona_phrase_bank,
            persona_speech_habits,
            persona_ng_phrases,
            persona_sample_lines,
        ],
        outputs=[persona_voice_dd, current_speaker],
    )

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

    demo.launch(server_name="0.0.0.0", server_port=port, share=False,
                allowed_paths=[str(OUTPUTS_DIR)])
