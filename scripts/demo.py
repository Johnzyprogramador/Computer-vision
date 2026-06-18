#!/usr/bin/env python3
from __future__ import annotations

import argparse

import gradio as gr
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Local browser demo for a trained YOLO model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)

    def predict(image, confidence):
        if image is None:
            return None, {"fire": 0.0, "smoke": 0.0}
        result = model.predict(np.asarray(image), conf=confidence, verbose=False)[0]
        scores = {"fire": 0.0, "smoke": 0.0}
        for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = str(model.names[int(class_id)]).lower()
            if name in scores:
                scores[name] = max(scores[name], float(score))
        # Ultralytics plots in BGR; Gradio expects RGB.
        return result.plot()[:, :, ::-1], scores

    with gr.Blocks(title="FireVision") as app:
        gr.Markdown(
            "# FireVision\nUpload an image to inspect fire/smoke boxes and confidence scores. "
            "This is not a certified alarm system."
        )
        with gr.Row():
            source = gr.Image(type="numpy", label="Input image")
            annotated = gr.Image(type="numpy", label="Detection")
        confidence = gr.Slider(0.01, 0.95, value=args.confidence, step=0.01, label="Confidence")
        scores = gr.Label(num_top_classes=2, label="Maximum class confidence")
        run = gr.Button("Detect", variant="primary")
        run.click(predict, inputs=[source, confidence], outputs=[annotated, scores])
        source.change(predict, inputs=[source, confidence], outputs=[annotated, scores])
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
