"""Gradio chat UI for NutriGuide."""

from __future__ import annotations

import logging

import gradio as gr

from nutriguide.pipeline import RagPipeline
from nutriguide.adaptation import compute_detail_level

CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
#header-title {
    text-align: center;
    font-size: 2.2rem !important;
    font-weight: 700;
    margin-bottom: 0;
    background: linear-gradient(90deg, #059669, #34d399);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
#header-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-top: 0.2em;
    margin-bottom: 1em;
}
#disclaimer-box {
    background: color-mix(in srgb, #f59e0b 12%, transparent);
    border: 1px solid color-mix(in srgb, #f59e0b 45%, transparent);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.88em;
}
#footer-note { text-align: center; opacity: 0.7; font-size: 0.8em; margin-top: 0.6em; }
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
    ["What snacks are safe for someone avoiding gluten?", "gluten-free"],
]


def build_demo(pipeline: RagPipeline) -> gr.Blocks:
    def chat_fn(message, history, constraints):
        response = pipeline.answer(message, history=history, constraints=constraints or "")
        level = compute_detail_level(history or [], message)
        return response, f"Response style: **{level.value.capitalize()}**"

    theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")
    with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="NutriGuide") as demo:
        gr.Markdown("# 🥗 NutriGuide", elem_id="header-title")
        gr.Markdown(
            "USDA-grounded dietary assistant — ask a nutrition question, get a sourced answer.",
            elem_id="header-subtitle",
        )
        gr.Markdown(DISCLAIMER, elem_id="disclaimer-box")

        constraints_box = gr.Textbox(
            label="Dietary constraints (optional)",
            placeholder="e.g., vegetarian, gluten-free, low-sodium",
        )

        style_label = gr.Markdown("Response style: **Standard**", elem_id="style-indicator")

        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[constraints_box],
            additional_outputs=[style_label],
            examples=EXAMPLES,
            type="messages",
            chatbot=gr.Chatbot(height=480, type="messages", label="NutriGuide"),
        )

        gr.Markdown(
            "Source: USDA Dietary Guidelines for Americans, 2020–2025 edition",
            elem_id="footer-note",
        )
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demo = build_demo(RagPipeline())
    demo.launch(share=True)


if __name__ == "__main__":
    main()
