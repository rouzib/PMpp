# Potential corrections

Potential-correction models live under `pmpp.corrections`. The module grouping separates reusable correction components from the core gravity and integration code.

Import corrections explicitly, for example:

```python
from pmpp.corrections import RadialPotentialCorrection
```

Use only correction classes that exist in the installed package; the documentation avoids inventing model features beyond the repository implementation.
