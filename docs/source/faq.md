# FAQ

## Why is the package called PM++ but imported as `pmpp`?

`PM++` is the display name. Python identifiers cannot contain `+`, so the distribution and import name are `pmpp`.

## Can I run without a GPU?

Small serial examples can run on CPU JAX. Multi-GPU examples require at least two visible GPU devices.

## Why does my first run take so long?

JAX is compiling fixed-shape computations.

## What does a capacity overflow mean?

A static-capacity buffer was too small for the particle or halo traffic in that run; increase the named capacity and rerun.

## Which parts are experimental?

Correction models, replicated/static mesh-halo variants, and some multi-GPU internals should be treated as research interfaces unless a release notes page says otherwise.
