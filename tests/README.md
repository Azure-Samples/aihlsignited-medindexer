# tests

usage:

```bash
pytest
```

alternatively,

```bash
export $(grep -v '^#' .env | xargs) && export PYTHONPATH="$(pwd):$PYTHONPATH" && export TARGET_DIRECTORY="." && pytest
```
