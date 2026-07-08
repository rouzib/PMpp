# Testing

Run the smallest targeted test first, then broaden coverage.

```bash
python -m pytest tests/test_grad_scatter.py -q
python -m pytest tests/test_grad_gather.py -q
python -m pytest tests/test_grad_gravity.py -q
python -m pytest tests/test_grad_nbody.py -q
```

Multi-GPU tests require suitable hardware and a CUDA-enabled JAX installation. CPU-only ReadTheDocs builds should not execute GPU tutorials or notebooks.
