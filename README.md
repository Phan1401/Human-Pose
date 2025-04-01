# LSTM-based Pose Classification Model Usage Guide

## Project Description
This project utilizes an LSTM model to classify human poses from skeleton data extracted using Mediapipe. The project consists of three main notebooks:

1. **create_data.ipynb**: Generates training data for the LSTM model.
2. **lstm_.ipynb**: Trains the LSTM model for pose classification.
3. **rc_lstm.ipynb**: Real-time pose recognition from webcam.

## Environment Setup
Required libraries:
```
pip install numpy pandas tensorflow keras mediapipe opencv-python
```

## Usage Guide
### Step 1: Data Preparation
- Pose data files should be in CSV format and contain skeleton frames extracted from Mediapipe.
- The pose labels include:
  - ngoi_lam_viec
  - ngoi_nga_lung
  - nam_ngu
  - gac_chan
  - dung_day
  - di_lai

### Step 2: Model Training
Run `lstm_.ipynb` to train the model:
- Ensure the CSV files are correctly located.
- The trained model will be automatically saved as `best_lstm_model.keras`.

### Step 3: Real-Time Pose Recognition
Run `rc_lstm.ipynb`:
- The webcam will be activated for pose recognition.
- Press `q` to exit.

## Contribution
For suggestions and improvements, please contact the development team via email.

## License
This project is licensed under the MIT License.
