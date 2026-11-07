"""Explainability (XAI) tools for the WAF agent.

features.py  - split an HTTP request into interpretable features and rebuild perturbed copies
probe.py     - one-forward-pass probability of the agent's decision (Llama, same prompt + tools)
lime.py      - LIME: perturb -> score -> weighted linear surrogate -> feature weights
report.py    - console table + HTML report
"""
