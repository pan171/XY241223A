# Settings

```sh
conda create -n pyqt python=3.10
source activate pyqt

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
pip install -r requirements.txt
```

# Package

```sh
pyinstaller --noconfirm --onefile --windowed --paths pages --add-data "img;img" --add-data "data;data" --add-data "pages;pages"  --hidden-import=matplotlib.backends.backend_pdf  --hidden-import=matplotlib.backends.backend_tkagg --hidden-import=matplotlib.backends.backend_agg  app.py
```