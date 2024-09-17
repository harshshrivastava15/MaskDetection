# Mask Detection Project

## Overview
This project aims to build a **mask detection system** that identifies whether a person is wearing a mask or not, using a Convolutional Neural Network (CNN) model. It was developed in Python using TensorFlow, OpenCV, and other machine learning libraries. The system is capable of real-time detection through a webcam or static image input.

## Project Structure
- **Dataset**: The dataset consists of two categories: `withMask` and `withoutMask`. Images are resized to 100x100 pixels for model input.
- **Model**: A CNN model built with TensorFlow to classify whether a person is wearing a mask.
- **Training**: The model is trained using a dataset of labeled images.
- **Evaluation**: The model's accuracy and loss are evaluated after training. Results are visualized using accuracy plots and a confusion matrix.
- **Real-time Detection**: The model is used to detect masks in real-time video streams from a webcam.

## Requirements
- Python 3.x
- Libraries:
  - `opencv-python`
  - `tensorflow`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `google-colab` (if running in Colab)

## Setup
1. Clone the project and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repo/mask-detection.git
   cd mask-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset by organizing images into `withMask` and `withoutMask` categories under the `data` folder.

## How to Run
1. **Training the Model**: Run the script `maskDetection.ipynb` to train the model.
   - The dataset is loaded and split into training and testing sets.
   - The model is built and trained on the dataset.
   - Results (accuracy and loss) are plotted for visualization.

2. **Real-time Mask Detection**:
   - The system uses OpenCV to capture video from the webcam.
   - Faces are detected, and the model predicts whether the detected face is masked or not.
   - The label (`Masked` or `Not Masked`) is displayed on the video feed.

   To start the real-time detection:
   ```bash
   python mask_detection_realtime.py
   ```

   Press `q` to exit the real-time detection.

3. **Static Image Detection**: You can detect masks on static images by calling the `detect_mask(image_path)` function in the script.

## Model Architecture
- Input Layer: 100x100x3 (Image size)
- Convolution Layers:
  - Conv2D (32 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D (pool size 2x2)
  - Conv2D (64 filters, kernel size 3x3, ReLU activation)
  - MaxPooling2D (pool size 2x2)
- Fully Connected Layers:
  - Dense (128 units, ReLU activation)
  - Dense (64 units, ReLU activation)
  - Output Layer: Dense (2 units, softmax activation)

## Results
- The model achieves a high accuracy in detecting masks with clear distinctions between masked and non-masked individuals.
- The results are visualized with accuracy and validation accuracy plots, as well as a confusion matrix.

## Future Improvements
- Adding more data to enhance model generalization.
- Fine-tuning the model to improve accuracy.
- Expanding the project for multi-class classification (e.g., improper mask usage).

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute.
