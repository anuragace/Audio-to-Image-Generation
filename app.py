import os
from functools import lru_cache

import gradio as gr
import numpy as np
import open_clip
import torch
from datasets import load_dataset
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import CLIPModel, CLIPProcessor, pipeline


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEFAULT_INDEX = os.getenv("DEFAULT_PINECONE_INDEX", "project-atoi-v2")
DEFAULT_DATASET = os.getenv("DEFAULT_KNOWLEDGE_DATASET", "rxc5667/3wordsdataset_noduplicates")
LORA_WEIGHTS_PATH = os.getenv("LORA_WEIGHTS_PATH", "rohith2812/atoi-lora-finetuned-v1")

INDEX_CHOICES = [
    "project-atoi-v2",
    "project-atoi",
]

DATASET_CHOICES = [
    "rohith2812/atoigeneration-final-data",
    "rxc5667/3wordsdataset_noduplicates",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is missing. Create a .env file from .env.example and add your key."
    )


pc = Pinecone(api_key=PINECONE_API_KEY)


@lru_cache(maxsize=2)
def get_dataset(dataset_name: str):
    return load_dataset(dataset_name, split="train")


@lru_cache(maxsize=1)
def get_clip_retrieval_model():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def get_depth_estimator():
    return pipeline("depth-estimation", device=0 if DEVICE == "cuda" else -1)


@lru_cache(maxsize=1)
def get_transcriber():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        device=0 if DEVICE == "cuda" else -1,
    )


@lru_cache(maxsize=1)
def get_generation_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=DTYPE,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=DTYPE,
    )

    if LORA_WEIGHTS_PATH:
        pipe.unet.load_attn_procs(LORA_WEIGHTS_PATH)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = pipe.to("cpu")

    return pipe


@lru_cache(maxsize=1)
def get_clip_score_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def retrieve_image_from_text_prompt(prompt: str, selected_index: str, knowledge_database: str):
    dataset = get_dataset(knowledge_database)
    index = pc.Index(selected_index)
    model, tokenizer = get_clip_retrieval_model()

    text_tokens = tokenizer([prompt]).to(DEVICE)
    with torch.no_grad():
        query_embedding = model.encode_text(text_tokens).cpu().numpy().flatten()

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=1,
        include_metadata=True,
        namespace="text_embeddings",
    )

    matches = getattr(results, "matches", None) or results.get("matches", [])
    if not matches:
        return None

    best_match = matches[0]
    metadata = getattr(best_match, "metadata", None) or best_match.get("metadata", {})
    image_path = metadata.get("image_path")
    description = metadata.get("description", "")

    if not image_path:
        return None

    for item in dataset:
        if item["image_path"].endswith(image_path):
            return {"image": item["image"], "description": description}

    return None


def get_depth_map(image):
    depth_image = get_depth_estimator()(image)["depth"]
    depth_array = np.array(depth_image)
    depth_array = depth_array[:, :, None]
    depth_array = np.concatenate([depth_array, depth_array, depth_array], axis=2)
    depth_tensor = torch.from_numpy(depth_array).float() / 255.0
    return depth_tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=DTYPE, device=DEVICE)


def calculate_clip_score(image, text: str):
    model, processor = get_clip_score_model()
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    return round(outputs.logits_per_image.item(), 4)


def transcribe_audio(audio):
    if audio is None:
        raise gr.Error("Please record or upload audio first.")

    sample_rate, waveform = audio
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude

    return get_transcriber()({"sampling_rate": sample_rate, "raw": waveform})["text"].strip()


def audio_to_image(audio, guidance_scale, num_inference_steps, selected_index, knowledge_database):
    transcription = transcribe_audio(audio)
    retrieved_data = retrieve_image_from_text_prompt(
        transcription,
        selected_index,
        knowledge_database,
    )

    if not retrieved_data:
        return transcription, None, None, "No relevant image found.", "N/A"

    retrieved_image = retrieved_data["image"]
    retrieved_description = retrieved_data["description"]
    depth_map = get_depth_map(retrieved_image)

    enhanced_image = get_generation_pipeline()(
        prompt=f"{transcription}. Ensure formulas are accurate and text is clean and legible.",
        image=retrieved_image,
        control_image=depth_map,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
    ).images[0]

    clip_score = calculate_clip_score(enhanced_image, transcription)
    return transcription, retrieved_image, enhanced_image, retrieved_description, clip_score


with gr.Blocks(title="Audio-to-Image Generation") as demo:
    gr.Markdown(
        """
        # Audio-to-Image Generation with AI
        Speak into the microphone to retrieve a related educational image, enhance it with ControlNet,
        and evaluate the result with CLIP.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="numpy", label="Speak Your Prompt")
            guidance_scale_input = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=8.5,
                label="Guidance Scale",
            )
            num_inference_steps_input = gr.Slider(
                minimum=10,
                maximum=200,
                step=10,
                value=100,
                label="Number of Inference Steps",
            )
            index_selection = gr.Dropdown(
                choices=INDEX_CHOICES,
                value=DEFAULT_INDEX,
                label="Pinecone Index",
            )
            knowledge_database_selection = gr.Dropdown(
                choices=DATASET_CHOICES,
                value=DEFAULT_DATASET,
                label="Knowledge Database",
            )
            submit_button = gr.Button("Generate Image", variant="primary")

        with gr.Column():
            transcription_output = gr.Textbox(label="Transcribed Prompt")
            retrieved_image_output = gr.Image(label="Retrieved Image")
            enhanced_image_output = gr.Image(label="Enhanced Image")
            retrieved_description_output = gr.Textbox(label="Retrieved Description")
            clip_score_output = gr.Textbox(label="CLIP Score")

    submit_button.click(
        fn=audio_to_image,
        inputs=[
            audio_input,
            guidance_scale_input,
            num_inference_steps_input,
            index_selection,
            knowledge_database_selection,
        ],
        outputs=[
            transcription_output,
            retrieved_image_output,
            enhanced_image_output,
            retrieved_description_output,
            clip_score_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
