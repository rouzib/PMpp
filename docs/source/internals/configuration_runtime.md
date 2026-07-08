# Configuration runtime

`Configuration` is the central binding between physical choices and runtime choices. It stores mesh spacing, particle grid shape, dtype, integration settings, optional correction configuration, multi-GPU runtime state, and derived arrays used throughout the solver.

`MultiGPUConfiguration` is nested under `Configuration(multigpu=...)` for the current API. It owns distributed runtime details such as compute mesh, communication permutations, halo metadata, capacity settings, and helper functions initialized for a particular decomposition. Legacy top-level compute-mesh paths should be treated as compatibility surfaces rather than the preferred style.

Because many fields influence static shapes or compiled control flow, changing a configuration often triggers recompilation. Keep configuration construction close to the script entry point and record it with simulation outputs.
