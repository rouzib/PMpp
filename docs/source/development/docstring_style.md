# Docstring style

Public docstrings should use NumPy style.

```python
def function_name(arg1, arg2, *, option=False):
    """One-sentence summary.

    Longer explanation of assumptions and shape conventions.

    Parameters
    ----------
    arg1 : type
        Description.
    arg2 : type
        Description.
    option : bool, optional
        Description.

    Returns
    -------
    result : type
        Description.

    Notes
    -----
    Mathematical or sharding convention.
    """
```

Avoid documenting private helpers as stable public API. Important private mechanisms can be described in internals pages instead.
