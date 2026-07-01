# Gradients

PM++ stages are written for JAX differentiation, and selected stages include custom adjoints to control memory use and scientific transposes. Start gradient experiments with the lightweight tests in `tests/` before scaling to multi-GPU cases.

Avoid differentiating large production runs until small meshes match expected finite-difference or PMWD-reference behavior.
