import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

    def _evaluate_classification(self, model, val_loader, history):
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
        self._plot_loss_curves(history, "PyTorch ANN Classifier")
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

    def _plot_loss_curves(self, history, title):
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='Training Loss Trend', color='skyblue')
        plt.plot(history['val_losses'], label='Validation Loss Trend', color='coral')
        plt.xlabel('Epochs', fontsize=12, fontweight='bold')
        plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
        plt.title(f'Evolution of Losses during Training Phase: {title}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        plt.savefig(f"{title.lower().replace(' ', '_')}_loss_curves.png")
        plt.close()

    def _plot_residual_plot(self, model, X_test, y_test, title):
        with torch.no_grad():
            predictions_on_test = model.forward(X_test).cpu().numpy()
            actual_on_test = y_test.cpu().numpy()
        residuals = actual_on_test - predictions_on_test
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions_on_test, residuals, color='forestgreen', alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Model Predictions')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title(f'Residual Plot: {title}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{title.lower().replace(' ', '_')}_residual_plot.png")
        plt.close()

    def _plot_joint_distribution(self, model, X_test, y_test, title):
        with torch.no_grad():
            y_test_pred = model.forward(X_test).cpu().numpy().flatten()
            y_test_act = y_test.cpu().numpy().flatten()
        joint_data = pd.DataFrame({'Actual': y_test_act, 'Predicted': y_test_pred})
        sns.jointplot(x='Actual', y='Predicted', data=joint_data, kind='reg', color='m')
        plt.suptitle(f'Model Prediction vs Actual Values: {title}', fontsize=14, fontweight='bold')
        plt.savefig(f"{title.lower().replace(' ', '_')}_joint_distribution.png")
        plt.close()

    def _plot_distribution_actual_predicted(self, model, X_test, y_test, title):
        with torch.no_grad():
            y_test_pred = model.forward(X_test).cpu().numpy().flatten()
            y_test_act = y_test.cpu().numpy().flatten()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(y_test_act, kde=True, color='darkorange')
        plt.title("Actual Values Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("House Prices", fontsize=12, fontweight='bold')
        plt.subplot(1, 2, 2)
        sns.histplot(y_test_pred, kde=True, color='darkviolet')
        plt.title("Model Predictions Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("House Prices", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}_distribution.png")
        plt.close()
