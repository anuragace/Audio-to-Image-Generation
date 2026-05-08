import os

import gradio as gr
from dotenv import load_dotenv

from retrieval import retrieve_image


load_dotenv()

DEFAULT_INDEX = os.getenv("DEFAULT_PINECONE_INDEX", "audio-to-image-sample")
DEFAULT_DATASET = os.getenv("DEFAULT_KNOWLEDGE_DATASET", "Anuragleo67/audio-to-image-sample-knowledge-base")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"


def search(prompt):
    if not prompt.strip():
        raise gr.Error("Enter a text prompt first.")

    result = retrieve_image(
        prompt=prompt,
        dataset_name=DEFAULT_DATASET,
        index_name=DEFAULT_INDEX,
    )
    if not result:
        return None, "No match found.", ""

    details = (
        f"Title: {result['title']}\n"
        f"Topic: {result['topic']}\n"
        f"Image path: {result['image_path']}\n"
        f"Score: {result['score']:.4f}"
    )
    return result["image"], result["description"], details


with gr.Blocks(title="Text Retrieval Demo") as demo:
    gr.Markdown("# Text-to-Image Retrieval")
    prompt_input = gr.Textbox(label="Text Prompt", value="line of best fit in linear regression")
    search_button = gr.Button("Search", variant="primary")
    image_output = gr.Image(label="Retrieved Image")
    description_output = gr.Textbox(label="Description")
    details_output = gr.Textbox(label="Match Details")

    search_button.click(
        fn=search,
        inputs=prompt_input,
        outputs=[image_output, description_output, details_output],
    )


if __name__ == "__main__":
    demo.launch(share=GRADIO_SHARE)
