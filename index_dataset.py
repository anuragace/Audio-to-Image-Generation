import argparse
import os

import open_clip
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


DEFAULT_NAMESPACE = "text_embeddings"
EMBEDDING_DIMENSION = 512


def create_embedding_model(device: str):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_texts(model, tokenizer, texts, device: str):
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy()


def ensure_index(pc: Pinecone, index_name: str, cloud: str, region: str):
    existing_indexes = {index["name"] for index in pc.list_indexes()}
    if index_name in existing_indexes:
        return

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Load a Hugging Face dataset, embed descriptions with CLIP, and upsert vectors to Pinecone."
    )
    parser.add_argument("--dataset", default=os.getenv("DEFAULT_KNOWLEDGE_DATASET"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", default=os.getenv("DEFAULT_PINECONE_INDEX", "audio-to-image-sample"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--create-index", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env, Colab secrets, or Kaggle secrets.")
    if not args.dataset:
        raise RuntimeError("Dataset is missing. Set DEFAULT_KNOWLEDGE_DATASET or pass --dataset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset(args.dataset, split=args.split)
    pc = Pinecone(api_key=api_key)

    if args.create_index:
        ensure_index(pc, args.index, args.cloud, args.region)

    index = pc.Index(args.index)
    model, tokenizer = create_embedding_model(device)

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
        texts = [
            item.get("embedding_text")
            or f"{item.get('topic', '')}. {item.get('title', '')}. {item['description']}"
            for item in batch
        ]
        embeddings = embed_texts(model, tokenizer, texts, device)

        vectors = []
        for item, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": item["id"],
                    "values": embedding.tolist(),
                    "metadata": {
                        "image_path": item["image_path"],
                        "description": item["description"],
                        "topic": item.get("topic", ""),
                        "title": item.get("title", ""),
                    },
                }
            )

        index.upsert(vectors=vectors, namespace=args.namespace)
        print(f"Upserted {start + len(batch)} / {len(dataset)} vectors")

    print(f"Done. Indexed {len(dataset)} records into '{args.index}' namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
