import argparse
import os

from dotenv import load_dotenv

from retrieval import search_pinecone


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test text prompt -> Pinecone search -> image_path.")
    parser.add_argument("prompt", nargs="?", default="line of best fit in linear regression")
    parser.add_argument("--index", default=os.getenv("DEFAULT_PINECONE_INDEX", "audio-to-image-sample"))
    parser.add_argument("--namespace", default="text_embeddings")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    matches = search_pinecone(
        args.prompt,
        index_name=args.index,
        namespace=args.namespace,
        top_k=args.top_k,
    )

    for rank, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        print(f"{rank}. id={match['id']}")
        print(f"   score={match['score']:.4f}")
        print(f"   image_path={metadata['image_path']}")
        print(f"   description={metadata.get('description', '')}")


if __name__ == "__main__":
    main()
