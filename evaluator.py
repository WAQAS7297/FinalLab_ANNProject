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

    def _evaluate_regression(self, model, X_val, y_val):
        with torch.no_grad():
            pred = model.forward(X_val)
            mse = mean_squared_error(y_val.numpy(), pred.numpy())
            mae = mean_absolute_error(y_val.numpy(), pred.numpy())
            r2 = r2_score(y_val.numpy(), pred.numpy())

        self._plot_regression_predictions(y_val.numpy(), pred.numpy())
        return mse, mae, r2

    def _evaluate_classification(self, model, val_loader):
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.view(val_x.size(0), -1)
                pred = model.forward(val_x)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.tolist())
                all_targets.extend(val_y.tolist())
        accuracy = accuracy_score(all_targets, all_preds) * 100
        cm = confusion_matrix(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

        self._plot_confusion_matrix(cm, "PyTorch ANN Classifier")
        return accuracy, cm, precision, recall, f1

    def _evaluate_keras_cnn(self, model, X_val, y_val, history):
        predictions = model.model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_val, predicted_classes) * 100
        cm = confusion_matrix(y_val, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, predicted_classes, average='weighted')

        self._plot_confusion_matrix(cm, "Keras CNN Classifier")
        self._plot_keras_training_history(history)
        return accuracy, cm, precision, recall, f1

    def _plot_regression_predictions(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Regression: Actual vs. Predicted Values")
        plt.grid(True)
        plt.savefig("regression_predictions.png")
        plt.close()

    def _plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {title}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(f"{title.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.close()

    def _plot_keras_training_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Keras CNN Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("keras_cnn_training_history.png")
        plt.close()