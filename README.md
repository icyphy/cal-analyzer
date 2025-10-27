# Reaction-level CAL Theorem

This repository stores LF programs used to develop a reaction-level CAL theorem, which refines the [original CAL theorem](https://dl.acm.org/doi/full/10.1145/3609119) in two ways:

1. Focusing on individual tasks rather than actors;
2. Reasoning about exact firing times over a concrete trace rather than bounds.

What's in this repo:

- `src/`: A set of LF programs used to generate ground-truth traces,
- `notebooks/`: Jupyter notebooks in either Python or TypeScript that use the reaction-level CAL theorem to predict traces, which should match the ground truth.