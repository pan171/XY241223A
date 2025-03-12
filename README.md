# Settings

```sh
conda create -n pyqt python=3.10
source activate pyqt

pip install -r requirements.txt
```

# Package

```sh
pyinstaller --noconfirm --onefile --windowed --paths pages --add-data "img;img" --add-data "data;data" --add-data "pages;pages"  --hidden-import=matplotlib.backends.backend_pdf  --hidden-import=matplotlib.backends.backend_tkagg --hidden-import=matplotlib.backends.backend_agg  app.py
```