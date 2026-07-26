# Citation

When PM++ is used to produce scientific results, cite the exact PM++ software
version/commit and the inherited adjoint-method paper. The software does not
currently declare an archival DOI, so include the repository URL and revision:

```bibtex
@software{pmpp,
  author  = {{PM++ developers}},
  title   = {{PM++}: Differentiable Multi-GPU Particle-Mesh Cosmology},
  year    = {2026},
  version = {replace-with-version-or-commit},
  url     = {https://github.com/rouzib/PMpp}
}
```

The simulation and discrete-adjoint foundation is described by
[Li et al., arXiv:2211.09815v2](https://arxiv.org/abs/2211.09815v2):

```bibtex
@misc{li2024differentiable,
  title         = {Differentiable Cosmological Simulation with the Adjoint Method},
  author        = {Yin Li and Chirag Modi and Drew Jamieson and Yucheng Zhang
                   and Libin Lu and Yu Feng and Fran\c{c}ois Lanusse
                   and Leslie Greengard},
  year          = {2024},
  eprint        = {2211.09815},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM}
}
```

Also cite the correction model, calibration data, and analysis method used by a
particular experiment. A PM++ citation alone does not describe those scientific
choices.
