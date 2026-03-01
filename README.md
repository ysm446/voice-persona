# Voice Persona

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を使ったテキスト読み上げ（TTS）Web UIアプリです。
Gradioベースのブラウザインターフェースから **Voice Clone** で音声を生成し、**Persona** で話者を管理できます。

## 機能

| モード | 説明 |
|---|---|
| **Voice Clone** | 参照音声をアップロードしてその声でテキストを読み上げ |
| **Persona** | 参照音声を登録・読み込み・削除して話者を管理 |

### 対応言語

Chinese / English / Japanese / Korean / German / French / Russian / Portuguese / Spanish / Italian / Auto

## セットアップ

### 必要環境

- Python 3.12
- NVIDIA GPU（CUDA対応）推奨
- conda

### インストール

```bash
# 1. conda環境のセットアップ（main環境にインストール）
conda activate main

# 2. PyTorch CUDA版をインストール（CUDAバージョンに合わせて変更）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. (オプション) FlashAttention 2（VRAM使用量削減）
pip install flash-attn --no-build-isolation
```

## 起動

### Windows（バッチファイル）

```
start.bat をダブルクリック
```

### コマンドライン

```bash
conda activate main
python app.py
```

起動後、ブラウザで http://localhost:7860 を開いてください。

> 各タブで初回「生成」クリック時にモデルが `models/` へダウンロードされます（1モデル約3〜4GB）。

## ファイル構成

```
voice-echo/
├── app.py            # メインアプリ（Gradio UI + モデル管理）
├── requirements.txt  # Python依存パッケージ
├── start.bat         # Windows起動スクリプト
├── models/           # モデルキャッシュ（自動生成・gitignore済み）
└── .gitignore
```

## 使用モデル

| モード | モデルID |
|---|---|
| Voice Clone | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
