"""Gradio side-by-side demo: base vs tuned (and eventually base/DPO/SimPO
three-way, Stage 10). See BUILD_PLAN.md section 8 (Stage 7).
"""
from __future__ import annotations


def build_app():
    raise NotImplementedError("Stage 7 - see BUILD_PLAN.md section 8")


if __name__ == "__main__":
    app = build_app()
    app.launch()
