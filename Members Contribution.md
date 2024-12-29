# Project Work Distribution

This project is a collaborative effort between **Member 1** and **Member B**, focused on implementing and comparing Neural Network models for regression and classification. This document details the specific tasks allocated to each member.

## Member 1 Responsibilities

1.  **PyTorch ANN Classification**
    *   Manages all aspects of the PyTorch ANN classification tasks within `main.py`.
        *   Loads data for classification using `DataLoaderModule('classification')`.
        *   Initializes the `ANN_Classifier` model.
        *   Trains the classification model using the `Trainer` module.
        *   Evaluates the classification model using the `Evaluator` module.

2.  **Visualization and Metrics Comparison**
    *   Implements and calls `plot_comparative_metrics()`.
    *   Creates comparative visualizations, including:
        *   Metrics comparison using bar charts.
        *   Training time comparison.

3.  **`DataLoader.py`**
    *   Handles data loading and preprocessing functionalities for classification tasks.

4.  **`Evaluator.py`**
    *   Focuses on the implementation of PyTorch ANN classification metrics and visualization of classification metrics.

5.  **`trainer.py`**
    *   Implements the training logic for the PyTorch ANN classification model.

6. **`model.py`**
    *   Focuses on high-level CNN architecture design, leveraging Keras's ease of use for implementation.

## Member B Responsibilities

1.  **PyTorch ANN Regression**
    *   Manages all aspects of the PyTorch ANN regression tasks within `main.py`.
        *   Loads the data for regression using `DataLoaderModule('regression')`.
        *   Initializes the `ANN_Regressor` model.
        *   Trains the regression model using the `Trainer` module.
        *   Evaluates the regression model using the `Evaluator` module.

2.  **Keras CNN Classification**
    *   Manages all aspects of the Keras CNN classification tasks within `main.py`.
        *   Loads the data for Keras CNN using `DataLoaderModule('keras_cifar100')`.
        *   Initializes the `KerasCNN` model.
        *   Trains the CNN model using the `Trainer` module.
        *   Evaluates the CNN model using the `Evaluator` module.

3.  **`DataLoader.py`**
    *   Handles data loading and preprocessing functionalities for Regression tasks and Keras CIFAR-100.

4.  **`Evaluator.py`**
    *   Focuses on the implementation of evaluation metrics for regression tasks and Keras CNN-related metrics.

5.  **`trainer.py`**
    *   Implements the training logic for both PyTorch ANN regression and Keras CNN models.

6.  **`model.py`**
    *   Works on foundational ANN implementations, focusing on low-level details such as gradient calculations.

## Summary Table

| Module         | Member 1                                                                     | Member B                                                                         |
| -------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `main.py`      | Manages PyTorch ANN Classification                                          | Manages PyTorch ANN Regression & Keras CNN Classification                         |
| `DataLoader.py`| Classification data loading                                                   | Regression and Keras CIFAR-100 data loading.                                       |
| `Evaluator.py`| PyTorch ANN classification metrics and visualization of classification metrics. | Regression metrics and Keras CNN-related metrics.                                  |
| `trainer.py`   | PyTorch ANN classification training.                                           | Regression and Keras CNN training.                                                 |
| `model.py`     | High-level Keras CNN architecture design.                                      | Foundational ANN implementation with focus on low-level details and gradient calculations.      |
| **Visualizations**   | Implementation and call for `plot_comparative_metrics`, including metrics comparison and training time analysis.                |                                                                                    |

This division of responsibilities ensures a balanced workload and fosters a comprehensive understanding of different aspects of the project. Each member is expected to maintain a clear communication and assist each other to achieve the project goals.