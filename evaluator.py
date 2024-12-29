import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    def __init__(self, model_type):
        self.model_type = model_type

    def evaluate(self, model, **kwargs):
        if self.model_type == 'ann_regressor':
            return self._evaluate_regression(model, **kwargs)
        elif self.model_type == 'ann_classifier':
            return self._evaluate_classification(model, **kwargs)
        elif self.model_type == 'keras_cnn':
            return self._evaluate_keras_cnn(model, **kwargs)
        else:
            raise ValueError("Unsupported model type.")

    def _evaluate_regression(self, model, X_val, y_val, X_test, y_test, history):
        with torch.no_grad():
            pred = model.forward(X_val)
            mse = mean_squared_error(y_val.numpy(), pred.numpy())
            mae = mean_absolute_error(y_val.numpy(), pred.numpy())
            r2 = r2_score(y_val.numpy(), pred.numpy())

        self._plot_regression_predictions(y_val.numpy(), pred.numpy())
        self._plot_loss_curves(history, "PyTorch ANN Regressor")
        self._plot_residual_plot(model, X_test, y_test, "PyTorch ANN Regressor")
        self._plot_joint_distribution(model, X_test, y_test, "PyTorch ANN Regressor")
        self._plot_distribution_actual_predicted(model, X_test, y_test, "PyTorch ANN Regressor")
        return mse, mae, r2

    def _evaluate_keras_cnn(self, model, X_val, y_val, history):
        predictions = model.model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_val, predicted_classes) * 100
        cm = confusion_matrix(y_val, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, predicted_classes, average='weighted')

        self._plot_confusion_matrix(cm, "Keras CNN Classifier")
        self._plot_keras_training_history(history)
        return accuracy, cm, precision, recall, f1

    
