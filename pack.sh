#/bin/bash

pyinstaller --noconfirm --onefile --windowed --paths src --add-data "img;img" --add-data "data;data" app.py

pyinstaller --noconfirm --onefile --windowed --paths pages --add-data "img;img" --add-data "data;data" --add-data "pages;pages" app.py
