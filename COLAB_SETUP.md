# Google Colab Setup

Use this guide after the latest repo changes have been pushed to GitHub.

## 1. Start A GPU Runtime

In Colab:

```text
Runtime > Change runtime type > T4 GPU
```

## 2. Clone The Repo

```python
!git clone https://github.com/anuragace/Audio-to-Image-Generation.git
%cd Audio-to-Image-Generation
```

## 3. Install Dependencies

```python
!pip install -U pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

If Colab asks you to restart the runtime after dependency installation, restart it, then run the setup cells again from the repo directory.

## 4. Add Environment Variables

Do not commit secrets to GitHub. In Colab, set them in a cell:

```python
import os

os.environ["PINECONE_API_KEY"] = "your-real-pinecone-key"
os.environ["DEFAULT_PINECONE_INDEX"] = "audio-to-image-sample"
os.environ["DEFAULT_KNOWLEDGE_DATASET"] = "Anuragleo67/audio-to-image-sample-knowledge-base"
os.environ["LORA_WEIGHTS_PATH"] = ""
os.environ["PINECONE_CLOUD"] = "aws"
os.environ["PINECONE_REGION"] = "us-east-1"
os.environ["GRADIO_SHARE"] = "true"
```

Keep `LORA_WEIGHTS_PATH` empty until the base app works.

## 5. Optional Hugging Face Login

The current sample dataset is public. Login is still useful if you later use private datasets, gated models, or your own LoRA.

```python
from huggingface_hub import login

login("your-hugging-face-token")
```

## 6. Verify Text Search

This confirms:

```text
text prompt -> CLIP embedding -> Pinecone -> image_path
```

```python
!python test_text_search.py "line of best fit in linear regression"
```

Expected top match:

```text
linear_regression_best_fit
images/linear_regression_best_fit.png
```

## 7. Run Text Retrieval UI

This confirms:

```text
prompt -> Pinecone -> Hugging Face image
```

```python
!python app_text_retrieval.py
```

Open the public Gradio URL printed by Colab.

## 8. Run Audio Retrieval UI

This confirms:

```text
audio -> Whisper -> text -> Pinecone -> Hugging Face image
```

```python
!python app_audio_retrieval.py
```

Open the public Gradio URL and allow microphone access.

## 9. Run Full ControlNet App

This final app does:

```text
audio -> Whisper -> Pinecone -> Hugging Face image -> depth map -> Stable Diffusion ControlNet -> enhanced image -> CLIP score
```

```python
!python app.py
```

The first run downloads several large models and may take a while.

## LoRA Plan

Do not use LoRA for the first Colab run.

Use:

```python
os.environ["LORA_WEIGHTS_PATH"] = ""
```

After retrieval, depth, and ControlNet generation work, you can replace it with your own Hugging Face LoRA:

```python
os.environ["LORA_WEIGHTS_PATH"] = "anuragace/your-lora-repo"
```

