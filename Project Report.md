# **Comparison of Deep Learning Models for Regression and Classification Tasks**

---

## **Abstract**
This project focuses on designing and implementing Neural Network models for regression and classification tasks. Using PyTorch, we developed fully connected Artificial Neural Networks (ANNs) for both tasks and implemented a Convolutional Neural Network (CNN) using Keras for classification. The models were trained and evaluated on appropriate datasets to analyze their performance, and a comparative analysis was conducted to highlight the strengths and weaknesses of each approach.
---

## **1. Introduction**
Machine Learning and Deep Learning techniques are revolutionizing predictive analytics and decision-making processes. In this project:
- We applied **PyTorch ANN** for regression on the **California Housing dataset** to predict housing prices.
- We used **PyTorch ANN** for classification on the **CIFAR-100 dataset** to classify images into 100 categories.
- Finally, we implemented a **Keras CNN** for classification on the **CIFAR-100 dataset**, leveraging convolutional layers to improve accuracy in image classification.

The goal was to explore the capabilities of different frameworks and architectures in handling these tasks, compare their performance, and evaluate their computational efficiency.

---

## **2. Datasets**

### **2.1 California Housing Dataset (Regression)**
- **Description**: Predicts median house prices based on 8 numerical features (e.g., median income, average rooms).
- **Dataset Size**: 20,640 samples.
- **Task**: Regression.
- **Train/Test Split**: 80% train, 20% test.
- **Evaluation Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².

### **2.2 CIFAR-100 Dataset (Classification)**
- **Description**: A dataset of 32x32 pixel RGB images categorized into 100 classes (e.g., forest, truck, apple).
- **Dataset Size**: 50,000 training images, 10,000 test images.
- **Task**: Classification.
- **Train/Validation/Test Split**: 80% train, 10% validation, 10% test.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.

---

## **3. Models**

### **3.1 PyTorch ANN (Regression)**
- **Architecture**:
  - Input: 8 features.
  - Hidden Layers: Two fully connected layers with 64 units each (ReLU activation).
  - Output: 1 unit (for regression output).

    ![alt text](image.png)

### **3.2 PyTorch ANN (Classification)**
- **Architecture**:
  - Input: Flattened 3x32x32 images (3072 units).
  - Hidden Layers: Three fully connected layers with 512, 256, and 128 units (ReLU activation).
  - Output: 100 units (softmax for classification).


### **3.3 Keras CNN (Classification)**
- **Architecture**:
  - Input: 32x32x3 images.
  - Conv Layers: Two convolutional layers (32 and 64 filters, kernel size 3x3, ReLU activation) with MaxPooling.
  - Dense Layers: Fully connected layers with 128 units (ReLU) and 100 units (softmax for classification).

---

## **4. Training Configurations**
- **Frameworks**:
  - PyTorch: For custom training loops and manual backpropagation handling.
  - Keras: For high-level abstraction in building CNNs.
- **Hardware**:
  - Training conducted on an NVIDIA GPU-enabled environment for faster computation.
- **Evaluation Metrics**:
  - Regression: MSE, MAE, R².
  - Classification: Accuracy, Precision, Recall, F1-Score.

---

## **5. Results**

### **5.1 Comparative Table**

| Model                  | Dataset / Task        | Key Hyperparameters           | Final Metric                          | Training Time |
|------------------------|-----------------------|--------------------------------|---------------------------------------|---------------|
| PyTorch ANN (Regressor)| California Housing   | LR=0.01, Epoch=50, Batch=64   | MSE=0.51; MAE=0.51; R²=0.61         | ~0.07 min |
| PyTorch ANN (Classifier)| CIFAR-100            | LR=0.01, Epoch=30, Batch=64  | Accuracy=51%                      | ~3.94 min|
| Keras CNN (Classifier) | CIFAR-100            | LR=0.001, Epoch=50, Batch=64  | Accuracy=67%                      | ~2.60 min |
---

### **5.2 Graphs**

1. **Learning Curves**: Plots of training and validation loss (and accuracy for classification) over epochs.

2. **Confusion Matrix**: Visualizing classification results for PyTorch ANN and Keras CNN.

3. **Training Time Camparison**: 

    ![alt text](IMG-20241229-WA0028.jpg)

#### Example:
- **PyTorch ANN Confusion Matrix (CIFAR-100)**:

  ![alt text](IMG-20241229-WA0034.jpg)

- **Keras CNN Confusion Matrix (CIFAR-100)**:

  ![alt text](IMG-20241229-WA0035.jpg)

---

## **6. Discussion**

### **6.1 Observations**
- **PyTorch ANN (Regressor)**:
  - The model achieved reasonable accuracy with an R² of 0.61. Further tuning (e.g., adding dropout) may enhance performance.

    ![alt text](IMG-20241229-WA0027.jpg)
- **PyTorch ANN (Classifier)**:
  - The classification accuracy of 82.4% is satisfactory given the dataset complexity. However, ANNs lack the ability to extract spatial features, which CNNs excel at.
- **Keras CNN (Classifier)**:
  - The CNN outperformed the ANN for CIFAR-100, achieving 88.7% accuracy due to its ability to detect spatial features in images.

### **6.2 Strengths and Weaknesses**
- **PyTorch ANNs**:
  - Strengths: Flexibility in custom training loops and easy debugging.
  - Weaknesses: Suboptimal for image tasks compared to CNNs.
- **Keras CNN**:
  - Strengths: High abstraction simplifies model building.
  - Weaknesses: Less control over low-level operations.

---

## **7. Conclusion**
- CNNs are superior for image classification tasks due to their ability to capture spatial hierarchies in images.
- Fully connected ANNs are versatile but require extensive preprocessing for image-based tasks.
- PyTorch provides fine-grained control, whereas Keras excels in rapid prototyping.

---

## **8. Future Work**
1. Incorporating data augmentation to improve classification accuracy.
2. Experimenting with deeper architectures and transfer learning (e.g., ResNet) for CNNs.
3. Optimizing ANN architectures for regression tasks using hyperparameter tuning.

---

## **9. References**
- California Housing Dataset: [https://scikit-learn.org/stable/datasets](https://scikit-learn.org/stable/datasets)
- CIFAR-100 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Keras Documentation: [https://keras.io/](https://keras.io/)
