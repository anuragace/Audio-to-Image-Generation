import argparse
import csv
import os
from pathlib import Path

import open_clip
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "sample_knowledge_base" / "train" / "metadata.csv"
DEFAULT_NAMESPACE = "text_embeddings"
EMBEDDING_DIMENSION = 512


def load_rows(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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

    parser = argparse.ArgumentParser(description="Index the sample knowledge-base dataset into Pinecone.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to metadata.csv.")
    parser.add_argument("--index", default=os.getenv("DEFAULT_PINECONE_INDEX", "audio-to-image-sample"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--create-index", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env or your notebook secrets.")

    metadata_path = Path(args.metadata)
    rows = load_rows(metadata_path)
    if not rows:
        raise RuntimeError(f"No rows found in {metadata_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = Pinecone(api_key=api_key)

    if args.create_index:
        ensure_index(pc, args.index, args.cloud, args.region)

    index = pc.Index(args.index)
    model, tokenizer = create_embedding_model(device)

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [row["embedding_text"] for row in batch]
        embeddings = embed_texts(model, tokenizer, texts, device)

        vectors = []
        for row, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": row["id"],
                    "values": embedding.tolist(),
                    "metadata": {
                        "image_path": row["image_path"],
                        "description": row["description"],
                        "topic": row["topic"],
                        "title": row["title"],
                    },
                }
            )

        index.upsert(vectors=vectors, namespace=args.namespace)
        print(f"Upserted {start + len(batch)} / {len(rows)} vectors")

    print(f"Done. Indexed {len(rows)} records into Pinecone index '{args.index}' namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
