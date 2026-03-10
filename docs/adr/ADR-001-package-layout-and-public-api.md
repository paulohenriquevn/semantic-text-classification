# ADR-001: Package Layout and Public API

**Status:** Accepted
**Date:** 2026-03-09
**Decision makers:** Architect Executor + NLP Engineering

## Context

The project needs a Python package structure that supports editable installs,
clean public imports, proper tooling (ruff, mypy, pytest), and long-term
extensibility across 9+ pipeline modules.

Two layout options were considered:

1. **`src/` as namespace**: `import src.models` — uses `src` as the top-level package
2. **`src/` as build layout**: `import talkex.models` — `src/` is invisible to consumers

## Decision

**Option 2: `src/` as build layout with `talkex` as the real package.**

```
src/
  talkex/
    __init__.py
    models/
    ingestion/
    segmentation/
    ...
```

Public imports:
```python
from talkex.models import Conversation
from talkex.exceptions import EngineError
```

## Consequences

### Positive
- `src` is never part of any import path — standard Python packaging practice
- `pip install -e .` works correctly with `[tool.setuptools.packages.find] where = ["src"]`
- Clean namespace: the package name matches the project identity
- ruff isort, mypy, and pytest all resolve the package correctly via `pythonpath = ["src"]`

### Negative
- Package name is long (`talkex`). Accepted trade-off: specificity > brevity.
- Every `__init__.py` must explicitly control re-exports to avoid accidental public API growth.

### Rules
- All public re-exports use explicit `__all__` or named imports in `__init__.py`
- Private symbols are prefixed with `_`
- No subpackage should be imported by consumers unless re-exported from its parent `__init__.py`
