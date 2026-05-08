# Audio-to-Image Generation with Retrieval and ControlNet

This project turns a spoken prompt into an image-enhancement workflow:

1. Transcribe the user's audio with Whisper.
2. Embed the transcription with OpenCLIP.
3. Search a Pinecone vector index for the most relevant image metadata.
4. Load the matched image from a Hugging Face dataset.
5. Estimate a depth map for that image.
6. Use Stable Diffusion + ControlNet + a LoRA adapter to generate an enhanced image.
7. Score the generated result against the transcription with CLIP.
8. Display everything in a Gradio UI.

## Why this exists

The goal is to create an audio-to-educational-image assistant. A user can say something like "explain line of best fit in linear regression", and the app tries to retrieve a related reference image from a knowledge dataset, then regenerate or polish that image so it better matches the spoken prompt.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your Pinecone API key:

```env
PINECONE_API_KEY=your-real-key
```

Run the app:

```powershell
python app.py
```

## Important security note

Never commit real API keys. If a key has already been pasted into chat, source code, or Git history, rotate it in Pinecone before using this project publicly.

## Hardware note

The generation pipeline is designed for CUDA GPUs. It can start on CPU, but Stable Diffusion image generation will be very slow and may run out of memory.

## Git workflow

Initialize and commit locally:

```powershell
git init
git add .
git commit -m "Initial audio-to-image generation app"
```

Push to GitHub after creating a remote repository:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```
