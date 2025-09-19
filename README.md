 # MNIST Digit Classifier with Deep Separable Convolutional Neural Network (DS-CNN)

This project implements a Deep Separable Convolutional Neural Network (DS-CNN) for robust digit recognition on the MNIST dataset. The model is designed to efficiently learn and classify handwritten digits, demonstrating high accuracy through a carefully constructed architecture and training regimen.
<img width="1024" height="1024" alt="Generated Image September 19, 2025 - 11_18PM" src="https://github.com/user-attachments/assets/a93ba2d9-1887-4f59-80a8-d98b813ab235" />

## Project Overview

The core of this project is a PyTorch-based DS-CNN model that processes grayscale images of handwritten digits. It leverages various advanced deep learning techniques, including convolutional layers, batch normalization, activation functions, pooling, and dropout, to achieve state-of-the-art performance in digit classification.

## Model Architecture: DS_CNN
<img width="1024" height="1024" alt="Generated Image September 19, 2025 - 11_06PM" src="https://github.com/user-attachments/assets/5b42f546-2e0a-4bb2-803a-657e241a0d0c" />

The `DS_CNN` model is a sequential deep learning network built with distinct blocks for feature extraction and classification. The architecture is designed to progressively reduce spatial dimensions while increasing feature complexity, culminating in a robust classification head.

**Total Trainable Parameters: 17,418**

Here's a breakdown of the layers and their functionalities:

### 1. Block 1: Initial Feature Extraction
<img width="1024" height="1024" alt="Generated Image September 19, 2025 - 11_11PM (1)" src="https://github.com/user-attachments/assets/a69db8b7-bc2a-4cea-b88e-a6c403b0dec8" />


This block focuses on extracting fundamental features from the input images.

*   **`conv1` (Convolutional Layer)**:
    *   **Type**: `nn.Conv2d`
    *   **Configuration**: `in_channels=1`, `out_channels=8`, `kernel_size=3`, `padding=0`
    *   **Purpose**: This layer takes the single-channel (grayscale) 28x28 MNIST image and applies 8 different 3x3 filters. It learns basic patterns like edges and corners.
    *   **Output Shape**: `26x26x8` (height, width, channels)
*   **`bn1` (Batch Normalization)**:
    *   **Type**: `nn.BatchNorm2d(8)`
    *   **Purpose**: Normalizes the activations from `conv1`. This helps stabilize and accelerate the training process by reducing internal covariate shift, allowing for higher learning rates.
*   **`conv2` (Convolutional Layer)**:
    *   **Type**: `nn.Conv2d`
    *   **Configuration**: `in_channels=8`, `out_channels=16`, `kernel_size=3`, `padding=0`
    *   **Purpose**: Further processes the 8 feature maps from `conv1` into 16 more complex feature maps.
    *   **Output Shape**: `24x24x16`
*   **`bn2` (Batch Normalization)**:
    *   **Type**: `nn.BatchNorm2d(16)`
    *   **Purpose**: Normalizes activations from `conv2`.
*   **`pool1` (Max Pooling Layer)**:
    *   **Type**: `nn.MaxPool2d`
    *   **Configuration**: `kernel_size=2`, `stride=2`
    *   **Purpose**: Reduces the spatial dimensions of the feature maps by half (from 24x24 to 12x12). This helps in making the model more robust to small translations in the input and reduces computational cost.
    *   **Output Shape**: `12x12x16`
*   **`dropout1` (Dropout Layer)**:
    *   **Type**: `nn.Dropout(0.1)`
    *   **Purpose**: Randomly sets 10% of the input features to zero during training. This regularization technique prevents overfitting by forcing the network to learn more robust features that are not reliant on specific neurons.

**Activation Function**: After each convolutional layer followed by batch normalization, a `nn.ReLU()` (Rectified Linear Unit) activation function is applied. ReLU introduces non-linearity, allowing the model to learn complex, non-linear relationships in the data.

### 2. Block 2: Deeper Feature Extraction
<img width="1024" height="1024" alt="Generated Image September 19, 2025 - 11_11PM (2)" src="https://github.com/user-attachments/assets/6b45a339-da51-44f9-8e19-35e9d1f75fd0" />

This block continues the process of feature learning, extracting even more abstract patterns.

*   **`conv3` (Convolutional Layer)**:
    *   **Type**: `nn.Conv2d`
    *   **Configuration**: `in_channels=16`, `out_channels=24`, `kernel_size=3`, `padding=0`
    *   **Purpose**: Processes the 16 feature maps from the previous block into 24 feature maps.
    *   **Output Shape**: `10x10x24`
*   **`bn3` (Batch Normalization)**:
    *   **Type**: `nn.BatchNorm2d(24)`
    *   **Purpose**: Normalizes activations from `conv3`.
*   **`conv4` (Convolutional Layer)**:
    *   **Type**: `nn.Conv2d`
    *   **Configuration**: `in_channels=24`, `out_channels=24`, `kernel_size=3`, `padding=0`
    *   **Purpose**: Further refines the 24 feature maps, maintaining the same number of channels but extracting different patterns.
    *   **Output Shape**: `8x8x24`
*   **`bn4` (Batch Normalization)**:
    *   **Type**: `nn.BatchNorm2d(24)`
    *   **Purpose**: Normalizes activations from `conv4`.
*   **`pool2` (Max Pooling Layer)**:
    *   **Type**: `nn.MaxPool2d`
    *   **Configuration**: `kernel_size=2`, `stride=2`
    *   **Purpose**: Reduces spatial dimensions further (from 8x8 to 4x4).
    *   **Output Shape**: `4x4x24`
*   **`dropout2` (Dropout Layer)**:
    *   **Type**: `nn.Dropout(0.2)`
    *   **Purpose**: Similar to `dropout1`, but with a higher dropout rate of 20%, indicating a need for stronger regularization at this deeper stage of feature extraction.

**Activation Function**: `nn.ReLU()` is applied after each convolutional layer followed by batch normalization.

### 3. Block 3: Classification Head
<img width="1024" height="1024" alt="Generated Image September 19, 2025 - 11_11PM" src="https://github.com/user-attachments/assets/ec8e3924-5417-40f5-a7e5-e01d731d4019" />


This final block is responsible for taking the highly abstract features and classifying them into one of the 10 digit classes.

*   **`conv5` (Convolutional Layer)**:
    *   **Type**: `nn.Conv2d`
    *   **Configuration**: `in_channels=24`, `out_channels=32`, `kernel_size=3`, `padding=0`
    *   **Purpose**: Transforms the 24 feature maps into 32 even more abstract feature maps.
    *   **Output Shape**: `2x2x32`
*   **`bn5` (Batch Normalization)**:
    *   **Type**: `nn.BatchNorm2d(32)`
    *   **Purpose**: Normalizes activations from `conv5`.
*   **`global_avg_pool` (Global Adaptive Average Pooling)**:
    *   **Type**: `nn.AdaptiveAvgPool2d((1, 1))`
    *   **Purpose**: This layer takes the `2x2x32` feature maps and reduces each 2x2 map to a single value, effectively converting the spatial features into a fixed-size vector of 32 features. This is a common technique to transition from convolutional layers to fully connected layers, making the model less sensitive to input image size variations.
    *   **Output**: 32 features
*   **`fc` (Fully Connected Layer)**:
    *   **Type**: `nn.Linear`
    *   **Configuration**: `in_features=32`, `out_features=10`
    *   **Purpose**: This is the final classification layer. It takes the 32 extracted features and maps them to 10 output classes, corresponding to the digits 0 through 9. The output of this layer represents the unnormalized scores (logits) for each class.

**Activation Function**: `nn.ReLU()` is applied after `conv5` followed by batch normalization.

## MNIST Dataset Utilization

The project utilizes the classic MNIST dataset, consisting of handwritten digits.

*   **Dataset Size**:
    *   **Training Dataset**: 50,000 images
    *   **Testing Dataset**: 10,000 images
*   **Data Preprocessing**:
    *   **`transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))`**: Applied to the training data to introduce random rotations (up to 10 degrees) and translations (up to 10% of the image width/height). This augments the training data, making the model more robust to variations in handwritten digits.
    *   **`transforms.ToTensor()`**: Converts the PIL Image format (default for MNIST) to a PyTorch `Tensor`.
    *   **`transforms.Normalize(mnist_mean, mnist_std)`**: Normalizes the pixel values of the images using the pre-calculated mean (0.1307) and standard deviation (0.3081) of the MNIST dataset. Normalization helps in faster convergence and better performance of the neural network.

### How Image Recognition Takes Place

1.  **Input**: A 28x28 grayscale image of a handwritten digit is fed into the `DS_CNN` model.
2.  **Feature Extraction (Blocks 1 & 2)**: The image sequentially passes through multiple convolutional layers, batch normalization, ReLU activations, max-pooling, and dropout layers.
    *   The convolutional layers learn to detect increasingly complex features (e.g., lines, curves, loops).
    *   Batch normalization ensures stable training.
    *   ReLU introduces non-linearity.
    *   Max pooling progressively reduces the spatial dimensions, making feature detection invariant to small shifts.
    *   Dropout regularizes the model to prevent overfitting.
3.  **Classification (Block 3)**:
    *   The final convolutional layer further refines features.
    *   Global Adaptive Average Pooling condenses the spatial information into a fixed-size feature vector.
    *   This feature vector is then fed into a fully connected layer.
    *   The fully connected layer outputs 10 values, each corresponding to the likelihood of the input image belonging to a specific digit class (0-9).
4.  **Prediction**: During inference, the digit class with the highest likelihood (highest output value from the fully connected layer) is chosen as the model's prediction.

## Training Details

*   **Optimizer**: Adam
*   **Initial Learning Rate**: `0.03`
*   **Loss Function**: `nn.CrossEntropyLoss()` (suitable for multi-class classification)
*   **Epochs**: 19
*   **Batch Size**: 32
*   **Device**: Training is performed on `cuda` (GPU) if available, otherwise `cpu`.
*   **Learning Rate Scheduler**: `optim.lr_scheduler.ReduceLROnPlateau` is used. This scheduler monitors the `test_acc` (test accuracy) and reduces the learning rate by a factor of 0.5 if the `test_acc` does not improve for `patience=0` epochs. This adaptive learning rate strategy helps in fine-tuning the model in later stages of training.

## Model Performance and Output

The model demonstrates excellent performance on the MNIST dataset. The training logs show a consistent improvement in both training and testing accuracy over the epochs.

### Training Progress (Sample Output)
<img width="878" height="470" alt="Screenshot 2025-09-19 233917" src="https://github.com/user-attachments/assets/6b4ccf3a-04bd-4239-a109-045bf2cd8d05" />

Epoch 1: Train Acc: 92.50% (Train set: 50000), Test Acc: 98.51% (Test set: 10000)
Epoch 2: Train Acc: 95.91% (Train set: 50000), Test Acc: 98.86% (Test set: 10000)
Epoch 3: Train Acc: 96.38% (Train set: 50000), Test Acc: 98.53% (Test set: 10000)
Epoch 4: Train Acc: 97.46% (Train set: 50000), Test Acc: 99.15% (Test set: 10000)
Epoch 5: Train Acc: 97.59% (Train set: 50000), Test Acc: 99.17% (Test set: 10000)
Epoch 6: Train Acc: 97.77% (Train set: 50000), Test Acc: 99.22% (Test set: 10000)
Epoch 7: Train Acc: 97.71% (Train set: 50000), Test Acc: 99.05% (Test set: 10000)
Epoch 8: Train Acc: 98.14% (Train set: 50000), Test Acc: 99.40% (Test set: 10000)
Epoch 9: Train Acc: 98.23% (Train set: 50000), Test Acc: 99.43% (Test set: 10000)
Epoch 10: Train Acc: 98.36% (Train set: 50000), Test Acc: 99.49% (Test set: 10000)
Epoch 11: Train Acc: 98.27% (Train set: 50000), Test Acc: 99.32% (Test set: 10000)
Epoch 12: Train Acc: 98.50% (Train set: 50000), Test Acc: 99.56% (Test set: 10000)
Epoch 13: Train Acc: 98.59% (Train set: 50000), Test Acc: 99.56% (Test set: 10000)
Epoch 14: Train Acc: 98.63% (Train set: 50000), Test Acc: 99.58% (Test set: 10000)
Epoch 15: Train Acc: 98.67% (Train set: 50000), Test Acc: 99.53% (Test set: 10000)
Epoch 16: Train Acc: 98.69% (Train set: 50000), Test Acc: 99.56% (Test set: 10000)
Epoch 17: Train Acc: 98.74% (Train set: 50000), Test Acc: 99.58% (Test set: 10000)
Epoch 18: Train Acc: 98.80% (Train set: 50000), Test Acc: 99.57% (Test set: 10000)
Epoch 19: Train Acc: 98.80% (Train set: 50000), Test Acc: 99.57% (Test set: 10000)
### Final Accuracy
<img width="859" height="547" alt="output" src="https://github.com/user-attachments/assets/e9c47437-d99a-47c0-8567-45a992652e36" />

After 19 epochs, the model achieved a remarkable **test accuracy of 99.57%** on the 10,000-image test set, with a training accuracy of 98.80% on the 50,000-image training set. This high accuracy indicates that the `DS_CNN` model effectively learned to distinguish between the different handwritten digits while generalizing well to unseen data.

## Usage

To run this project:

1.  Ensure you have Python and necessary libraries (PyTorch, torchvision, matplotlib, numpy) installed. It's recommended to use `uv` with `pyproject.toml` for environment management.
2.  Execute the `Session5_TorchModel.ipynb` Jupyter Notebook. The notebook will automatically download the MNIST dataset if it's not present in the `./data` directory.

## Visualizations

The `Session5_TorchModel.ipynb` notebook also includes code to plot the training and test accuracies over epochs, providing a clear visual representation of the model's learning progress.
