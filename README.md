# Alice & Bob x YQuantum 2026 Challenge

Challenge materials are in `challenge/`.

## Environment setup

This repository now includes a local Python `3.11` environment setup that works with the notebook in `challenge/1-challenge.ipynb`.

Recommended steps on Windows PowerShell:

```powershell
uv python install 3.11
uv venv --seed --python 3.11 .venv311
.\.venv311\Scripts\python -m pip install -r requirements.txt
```

After installation:

1. Open `challenge/1-challenge.ipynb`.
2. Select the interpreter or kernel from `.venv311`.
3. Run the import cells at the top of the notebook.

## Notes

- The notebook metadata shows it was originally run in Python `3.11.15`.
- This repository is now configured to use a local Python `3.11` environment at `.venv311`.
- The direct notebook dependencies are listed in `requirements.txt`.
