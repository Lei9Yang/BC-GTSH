"""Built-in method registration."""

_loaded = False


def load_builtin_methods() -> None:
    global _loaded
    if _loaded:
        return
    from . import bc_gtsh as _bc_gtsh  # noqa: F401

    _loaded = True


__all__ = ["load_builtin_methods"]

