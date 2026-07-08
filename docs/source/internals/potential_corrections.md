# Potential corrections

Potential-correction code lives under `pmpp.corrections`, with compatibility imports in `pmpp.potential_correction` where available. Correction families include radial/window utilities, softening-related models, particle-grid-deconvolution helpers, mesh-CNN components, and combined wrappers.

Correction models are experimental unless a release explicitly marks them stable. Treat a correction as part of the scientific model: record parameters, training data if applicable, validation target, and expected tolerance.
