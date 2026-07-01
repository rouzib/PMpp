# mesh_halo mode

`mesh_halo` is the preferred multi-GPU mode because halo communication follows the mesh decomposition used by the Particle-Mesh force solve. It reduces ambiguity around which device owns mesh cells and which particles must be visible to neighboring subdomains.

Use this mode first for new multi-GPU simulations unless you have a specific reason to test another communication strategy.
