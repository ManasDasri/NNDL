---
title: Contract Clause Risk Detection
emoji: 📑
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Contract clause risk detection

Flags six risk-relevant clause types in commercial contracts: Cap on Liability,
Non-Compete, License Grant, Audit Rights, Termination for Convenience and
Insurance. The model is the project's TextCNN baseline, trained on
[CUAD v1](https://www.atticusprojectai.org/cuad), and runs on a free CPU.

Paste a contract and press **Analyse**. You get a score per clause type, the
passages that triggered each flag, and the phrases the model relied on.

- Results for all four models: https://www.algorithmicbit.tech/NNDL/
- Code and write-up: https://github.com/ManasDasri/NNDL
- Model: https://huggingface.co/KenX049/cuad-clause-risk-cnn

A research project, not legal advice.
