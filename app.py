import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
from rag import answer_query

CUSTOM_CSS = """
#header-title {
    text-align: center;
    font-size: 2rem !important;
    margin-bottom: 0.2em;
}
#header-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 1em;
}
#disclaimer-box {
    background: #fff8e1;
    border: 1px solid #f0c14b;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.9em;
    color: #6b5500;
}
.dark #disclaimer-box {
    background: #3a2f00;
    border: 1px solid #7a6100;
    color: #ffd966;
}
"""

DISCLAIMER = (
    "⚠️ **Educational prototype.** NutriGuide provides general dietary information "
    "grounded in the USDA Dietary Guidelines for Americans. It does not provide "
    "personalized medical or nutritional advice. Consult a registered dietitian "
    "or physician for guidance specific to your health needs."
)

EXAMPLES = [
    ["How much saturated fat should I eat daily?", ""],
    ["What are good protein sources for a vegetarian diet?", "vegetarian"],
    ["What is the recommended daily sodium intake?", ""],
    ["How many calories should an adult eat per day?", ""],
    ["What foods should I limit on a low-sodium diet?", "low-sodium"],
    ["What's a healthy plate look like for dinner?", ""],
    ["How much added sugar is recommended per day?", ""],
    ["What are good sources of fiber?", "vegetarian"],
]


def chat_fn(message, history, constraints):
    return answer_query(message, constraints)


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# 🥗 NutriGuide", elem_id="header-title")
    gr.Markdown(
        "USDA-grounded dietary assistant — ask a nutrition question, get a sourced answer.",
        elem_id="header-subtitle",
    )
    gr.Markdown(DISCLAIMER, elem_id="disclaimer-box")

    with gr.Row():
        constraints_box = gr.Textbox(
            label="Dietary constraints (optional)",
            placeholder="e.g., vegetarian, gluten-free, low-sodium",
            scale=1,
        )

    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[constraints_box],
        examples=EXAMPLES,
        type="messages",
        chatbot=gr.Chatbot(
            height=480,
            avatar_images=(None, "🥗"),
            type="messages",
        ),
    )

    gr.Markdown(
        "<center><sub>Source: USDA Dietary Guidelines for Americans, 2020–2025 edition</sub></center>"
    )

if __name__ == "__main__":
    demo.launch(share=True)