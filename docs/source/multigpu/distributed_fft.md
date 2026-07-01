# Distributed FFT support

Distributed FFT helpers construct transforms for sharded meshes. At a high level, PM++ arranges data so mesh fields can move between real-space decomposition and Fourier-space operations needed by the force solve.

The implementation is in `pmpp.FFT_distributed` and related mesh/FFT utilities. It should be treated as runtime infrastructure: examples in these docs describe its role but do not execute multi-GPU FFTs on Read the Docs.
