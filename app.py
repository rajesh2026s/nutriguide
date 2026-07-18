import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
from rag import answer_query

DESCRIPTION = (
    "Ask a question about nutrition or meal planning, grounded in the USDA "
    "Dietary Guidelines for Americans. Optionally add dietary constraints "
    "(e.g., vegetarian, low-sodium) to tailor the answer. This tool provides "
    "general dietary information only and is not personalized medical advice."
)


def chat_fn(message, history, constraints):
    return answer_query(message, constraints)


with gr.Blocks() as demo:
    gr.Markdown("## NutriGuide: USDA-Grounded Dietary Assistant")
    gr.Markdown(DESCRIPTION)

    constraints_box = gr.Textbox(
        label="Dietary constraints (optional)",
        placeholder="e.g., vegetarian, gluten-free, low-sodium",
    )

    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[constraints_box],
        examples=[
            ["How much saturated fat should I eat daily?", ""],
            ["What are good protein sources for a vegetarian diet?", "vegetarian"],
            ["What is the recommended daily sodium intake?", ""],
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)