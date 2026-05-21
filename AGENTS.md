# Agents

## Cursor Cloud specific instructions

### Overview

This is a Python data science project analyzing data from the **Divar** online classifieds platform. It consists of 5 Jupyter notebooks in `src/` covering EDA, statistical analysis, a recommender system, and ML price/credit prediction. See `README.md` for the full project description.

### Running notebooks

- Start Jupyter Lab: `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="" --allow-root`
- Execute a notebook headlessly: `jupyter nbconvert --to notebook --execute src/<notebook>.ipynb --ExecutePreprocessor.timeout=300`
- Ensure `$HOME/.local/bin` is on `PATH` (pip installs Jupyter CLI there).

### Dataset

The Divar CSV dataset (~1M rows, 64 columns) is **not included** in the repository (gitignored). Notebooks expect it at `../Divar Dataset/Divar.csv` (relative to `src/`), except `ml_prediction_price.ipynb` which loads `./Divar.csv` (relative to `src/`). To run notebooks end-to-end, place the dataset at both locations or create a synthetic one for testing.

### Key gotchas

- **No `requirements.txt` in upstream**: The README references one, but it was never committed. A `requirements.txt` was added to capture all notebook dependencies.
- **Python 3.12 works**: Notebooks target Python 3.11 per the README badge, but all libraries work on the pre-installed Python 3.12.
- **matplotlib backend**: When running headlessly (no display), set `matplotlib.use('Agg')` before importing pyplot, or use `MPLBACKEND=Agg` env var.
- **jdatetime version attribute**: Use `jdatetime.__VERSION__` (uppercase), not `jdatetime.__version__`.
- **Pandas 3.x warnings**: Some notebooks may emit `Pandas4Warning` about `select_dtypes` with `'object'` dtype — these are deprecation warnings and do not affect execution.
