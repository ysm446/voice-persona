# voice-echo

Qwen3-TTSを使ったTTS Web UIアプリ。GradioベースのブラウザUI。

## 起動

```bash
conda activate main
python app.py
# → http://localhost:7860
```

Windowsでは `start.bat` をダブルクリックでも起動可能。

## 構成

- `app.py` — メインアプリ。モデル管理（遅延ロード）＋Gradio UI（3タブ）
- `requirements.txt` — 依存パッケージ
- `start.bat` — Windows起動スクリプト（conda env: main）
- `models/` — HuggingFaceモデルキャッシュ（gitignore済み）

## 重要な実装メモ

### HF_HOME の設定順序
`app.py` の先頭で `os.environ["HF_HOME"]` を設定している。
これは `transformers` / `qwen_tts` の import **より前** に置く必要がある。
順序を変えるとモデルが `models/` ではなく `~/.cache/huggingface/` に保存される。

### 遅延モデルロード
3モデルを同時ロードすると約24GB VRAMを消費するため、`get_model(key)` で初回使用時のみロードする。
ロード済みモデルは `_models` dictにキャッシュし、セッション中は保持し続ける。

### Voice Clone の引数順序
Gradio `gr.Audio(type="numpy")` は `(sample_rate, numpy_array)` で渡してくるが、
`qwen_tts` の `ref_audio` 引数は `(numpy_array, sample_rate)` を期待する（逆順）。
`generate_clone` 内で `ref_sr, ref_data = ref_audio` → `(ref_data, ref_sr)` と入れ替えている。

## モデル

| キー | モデルID |
|---|---|
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `voice_clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

## 依存パッケージのインストール

```bash
# PyTorch CUDA版（CUDAバージョンに合わせて変更）
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# オプション: FlashAttention 2
pip install flash-attn --no-build-isolation
```
