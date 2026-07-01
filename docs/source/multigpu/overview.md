# Multi-GPU overview

Multi-GPU runs use `pmpp.multigpu_configuration.MultiGPUConfiguration` nested inside `pmpp.configuration.Configuration`. A compute mesh is usually built from JAX devices with `pmpp.utils.create_compute_mesh`.

The preferred production mode is `mesh_halo`, which keeps ownership and halo exchange structured around mesh slabs. Multi-GPU examples require actual multi-GPU hardware and are not executed by the documentation build.
