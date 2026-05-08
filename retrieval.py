import os
from functools import lru_cache

import open_clip
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

DEFAULT_NAMESPACE = "text_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")
    return Pinecone(api_key=api_key)


@lru_cache(maxsize=1)
def get_clip_model():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=4)
def get_dataset(dataset_name: str):
    if os.path.isdir(dataset_name):
        return load_dataset("imagefolder", data_dir=dataset_name, split="train")
    return load_dataset(dataset_name, split="train")


@lru_cache(maxsize=4)
def get_image_lookup(dataset_name: str):
    dataset = get_dataset(dataset_name)
    return {item["image_path"]: item for item in dataset}


def embed_prompt(prompt: str):
    model, tokenizer = get_clip_model()
    tokens = tokenizer([prompt]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().tolist()


def search_pinecone(prompt: str, index_name: str, namespace: str = DEFAULT_NAMESPACE, top_k: int = 3):
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    result = index.query(
        vector=embed_prompt(prompt),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return result["matches"]


def retrieve_image(prompt: str, dataset_name: str, index_name: str, namespace: str = DEFAULT_NAMESPACE):
    matches = search_pinecone(prompt, index_name=index_name, namespace=namespace, top_k=1)
    if not matches:
        return None

    match = matches[0]
    metadata = match["metadata"]
    image_path = metadata["image_path"]
    item = get_image_lookup(dataset_name).get(image_path)
    if not item:
        return None

    return {
        "image": item["image"],
        "image_path": image_path,
        "description": metadata.get("description", item.get("description", "")),
        "title": metadata.get("title", item.get("title", "")),
        "topic": metadata.get("topic", item.get("topic", "")),
        "score": match["score"],
    }
