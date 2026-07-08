# Architecture

```text
white noise -> linear modes -> LPT -> particles
particles -> scatter -> FFT Poisson solve -> gather -> drift/kick loop
loss/observable -> custom VJP/adjoint -> gradients
```

The active implementation lives in `src/pmpp`. The `pmwd/` directory is retained as a reference implementation for validation and comparison work.
