import os

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline

from retrieval import retrieve_image


load_dotenv()

DEFAULT_INDEX = os.getenv("DEFAULT_PINECONE_INDEX", "audio-to-image-sample")
DEFAULT_DATASET = os.getenv("DEFAULT_KNOWLEDGE_DATASET", "Anuragleo67/audio-to-image-sample-knowledge-base")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe(audio):
    if audio is None:
        raise gr.Error("Record or upload audio first.")

    sample_rate, waveform = audio
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude
    return transcriber({"sampling_rate": sample_rate, "raw": waveform})["text"].strip()


def search_from_audio(audio):
    prompt = transcribe(audio)
    result = retrieve_image(
        prompt=prompt,
        dataset_name=DEFAULT_DATASET,
        index_name=DEFAULT_INDEX,
    )
    if not result:
        return prompt, None, "No match found.", ""

    details = (
        f"Title: {result['title']}\n"
        f"Topic: {result['topic']}\n"
        f"Image path: {result['image_path']}\n"
        f"Score: {result['score']:.4f}"
    )
    return prompt, result["image"], result["description"], details


with gr.Blocks(title="Audio Retrieval Demo") as demo:
    gr.Markdown("# Audio-to-Image Retrieval")
    audio_input = gr.Audio(type="numpy", label="Speak Your Prompt")
    search_button = gr.Button("Search", variant="primary")
    transcription_output = gr.Textbox(label="Transcription")
    image_output = gr.Image(label="Retrieved Image")
    description_output = gr.Textbox(label="Description")
    details_output = gr.Textbox(label="Match Details")

    search_button.click(
        fn=search_from_audio,
        inputs=audio_input,
        outputs=[transcription_output, image_output, description_output, details_output],
    )


if __name__ == "__main__":
    demo.launch(share=GRADIO_SHARE)
