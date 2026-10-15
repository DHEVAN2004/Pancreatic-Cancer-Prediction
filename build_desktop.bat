@echo off
REM Build a standalone Windows executable for the desktop app using PyInstaller.

REM 1) Install PyInstaller if missing
python -m pip install --upgrade pip
python -m pip install pyinstaller

REM 2) Build the exe (bundles model and scaler files into the dist folder)
REM    Adjust the paths below if your files are in a different location.
python -m PyInstaller --noconfirm --onefile --windowed ^
  --name PancreaticCancerPredictor ^
  --add-data "pancreatic_knn_model.pkl;." ^
  --add-data "scaler.pkl;." ^
  --add-data "model_metadata.pkl;." ^
  pancreatic_cancer_prediction_app.py

REM 3) The generated .exe will be in the dist\PancreaticCancerPredictor.exe
if exist dist\PancreaticCancerPredictor.exe (
  echo Build complete! You can run dist\PancreaticCancerPredictor.exe
) else (
  echo Build failed. Check the PyInstaller output for errors.
)
