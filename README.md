# Skin-Cancer_Detection
# Skin Cancer Detection using CNN 🔬

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-CNN-red.svg)

Deep Learning model for detecting 7 types of skin cancer using Convolutional Neural Networks (CNN) trained on the HAM10000 dataset.

## Overview

This project implements a CNN-based image classification system to identify different types of skin lesions. The model is trained on 28x28 RGB images and can classify lesions into 7 categories with high accuracy.

---

## Skin Cancer Types Detected

The model classifies skin lesions into 7 categories:

| Label | Code | Description |
|-------|------|-------------|
| 0 | akiec | Actinic keratoses and intraepithelial carcinoma |
| 1 | bcc | Basal cell carcinoma |
| 2 | bkl | Benign keratosis-like lesions |
| 3 | df | Dermatofibroma |
| 4 | nv | Melanocytic nevi |
| 5 | vasc | Pyogenic granulomas and hemorrhage |
| 6 | mel | Melanoma |

---

## Features

- ✅ **Data Preprocessing**: Image normalization and reshaping
- ✅ **Data Balancing**: RandomOverSampler to handle class imbalance
- ✅ **CNN Architecture**: 3 convolutional layers with max pooling
- ✅ **Activation**: Swish activation function for better performance
- ✅ **Regularization**: Dropout layer to prevent overfitting
- ✅ **Callbacks**: ModelCheckpoint and EarlyStopping for optimal training
- ✅ **Visualization**: Training/validation loss and accuracy plots

---

## Model Architecture
```
Input (28x28x3)
    ↓
Conv2D (32 filters, 2x2) + Swish + MaxPooling
    ↓
Conv2D (32 filters, 2x2) + Swish + MaxPooling
    ↓
Conv2D (64 filters, 2x2) + Swish + MaxPooling
    ↓
Flatten
    ↓
Dense (64) + Swish
    ↓
Dropout (0.5)
    ↓
Dense (7) + Softmax
```

**Optimizer**: Nadam  
**Loss Function**: Sparse Categorical Crossentropy  
**Metrics**: Accuracy

---

## Dataset

**HAM10000 Dataset** (Human Against Machine with 10000 training images)
- **Images**: 28x28 RGB format
- **Total Samples**: ~10,000 (after oversampling)
- **Classes**: 7 types of skin lesions

### Data Files Required:
1. `hmnist_28_28_RGB.csv` - Image data in CSV format
2. `HAM10000_metadata.csv` - Metadata about the lesions

---

## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow/Keras
- Required libraries

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/Wannakorn-Sangthongngam/Skin-Cancer_Detection.git
   cd Skin-Cancer_Detection
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download the dataset**
   - Download HAM10000 dataset
   - Place `hmnist_28_28_RGB.csv` and `HAM10000_metadata.csv` in the project directory

---

## Usage

### Training the Model
```python
# Run the Jupyter notebook
jupyter notebook model.ipynb
```

Or run the Python script:
```python
python model.py
```

### Model Training Process

1. **Load Data**: Reads CSV files containing image data
2. **Data Balancing**: Uses RandomOverSampler to balance classes
3. **Preprocessing**: Normalizes pixel values (0-255 → 0-1)
4. **Train/Test Split**: 80/20 split
5. **Model Training**: 100 epochs with early stopping
6. **Model Saving**: Best model saved as `skin.h5`

### Making Predictions
```python
from keras.models import load_model
import numpy as np

# Load trained model
model = load_model('skin.h5')

# Prepare your image (28x28x3)
image = np.array(your_image).reshape(1, 28, 28, 3) / 255

# Predict
prediction = model.predict(image)
class_id = np.argmax(prediction)
```

---

## Results

The model achieves:
- **Training Accuracy**: ~XX%
- **Validation Accuracy**: ~XX%
- **Training Loss**: ~X.XX
- **Validation Loss**: ~X.XX

*(Update with your actual results)*

### Training Visualizations

The notebook generates:
- Loss curve (Training vs Validation)
- Accuracy curve (Training vs Validation)

---

## Project Structure
```
Skin-Cancer_Detection/
├── model.ipynb              # Main Jupyter notebook
├── skin.h5                  # Trained model (generated)
├── hmnist_28_28_RGB.csv    # Image dataset
├── HAM10000_metadata.csv   # Metadata
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## Dependencies
```txt
numpy
pandas
tensorflow
keras
scikit-learn
imbalanced-learn
matplotlib
```

Create `requirements.txt`:
```bash
cat > requirements.txt << EOF
numpy>=1.19.0
pandas>=1.1.0
tensorflow>=2.6.0
keras>=2.6.0
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
matplotlib>=3.3.0
EOF
```

---

## Key Techniques Used

### 1. **Data Imbalance Handling**
- **Problem**: Highly imbalanced dataset (nv class dominates)
- **Solution**: RandomOverSampler to balance all classes

### 2. **CNN Architecture**
- Multiple convolutional layers for feature extraction
- MaxPooling for dimensionality reduction
- Swish activation for better gradient flow

### 3. **Regularization**
- Dropout (0.5) to prevent overfitting
- Early stopping to avoid overtraining

### 4. **Optimization**
- Nadam optimizer (Adam + Nesterov momentum)
- ModelCheckpoint to save best model
- Early stopping with patience=10

---

## Future Improvements

- [ ] Implement data augmentation for better generalization
- [ ] Try transfer learning (ResNet, EfficientNet)
- [ ] Add confusion matrix and classification report
- [ ] Deploy model as web application
- [ ] Test on higher resolution images
- [ ] Implement ensemble methods

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- HAM10000 Dataset creators
- TensorFlow and Keras teams
- Open source community

---

## Contact

**Your Name** - [Your GitHub](https://github.com/Wannakorn-Sangthongngam)

Project Link: [https://github.com/Wannakorn-Sangthongngam/Skin-Cancer_Detection](https://github.com/Wannakorn-Sangthongngam/Skin-Cancer_Detection)

---

## Citation

If you use this code, please cite:
```
@misc{skin_cancer_detection,
  author = {Wannakorn Sangthongngam},
  title = {Skin Cancer Detection using CNN},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Wannakorn-Sangthongngam/Skin-Cancer_Detection}
}
```

---

**⭐ If you find this project helpful, please give it a star!**
