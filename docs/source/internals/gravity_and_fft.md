# Gravity and FFT

The gravity path normalizes deposited density, forms density contrast (`dens - 1`), solves the Poisson equation in Fourier space, and gathers force components back to particles. Options such as particle-Nyquist filtering, interlacing, CIC compensation, and potential corrections modify this pipeline.

FFT layout is a runtime concern. Serial runs can use local FFTs, while multi-GPU runs require distributed transforms and layout transposes. The zero mode, spectral derivative conventions, and dtype all affect validation tolerances.

Potential corrections should be treated as explicit model choices. Record the correction family and parameters with any scientific output.
