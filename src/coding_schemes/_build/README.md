# Coding scheme build system

This folder holds the single source of truth for all six coding-scheme dossiers and the generator that renders them.

- `content.py` is the structured, enriched specification of every scheme. **Edit content here.**
- `assets.py` holds the shared CSS and JavaScript for the HTML surfaces.
- `build.py` renders each scheme to LaTeX (compiled to PDF), interactive HTML, and README, plus the aggregated `index.html` and the directory `README.md`.

## Build

```bash
python3 build.py            # render everything and compile PDFs with tectonic
python3 build.py --no-pdf   # render text surfaces only
```

PDF compilation uses `tectonic`. If it is not installed, run with `--no-pdf` and compile the `.tex` files with any LaTeX engine.
