from data_loader import DataLoaderModule
from models import ANN_Regressor, ANN_Classifier, KerasCNN
from trainer import Trainer
from evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def main():
    print("Starting PyTorch ANN Regression Training...")
    reg_input_size = 8
    reg_hidden_sizes = [64, 64]
    reg_output_size = 1
    reg_lr = 0.001
    reg_epochs = 50
    reg_batch_size = 64

    data_loader = DataLoaderModule('regression')
    X_train, y_train, X_val_reg, y_val_reg = data_loader.load_data()
    X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(X_val_reg, y_val_reg, test_size=0.5, random_state=42)
    reg_model = ANN_Regressor(reg_input_size, reg_hidden_sizes, reg_output_size)
    trainer = Trainer('ann_regressor')
    reg_model, reg_history, reg_time = trainer.train(reg_model, lr=reg_lr, epochs=reg_epochs, batch_size=reg_batch_size, X_train=X_train, y_train=y_train, X_val=X_val_reg, y_val=y_val_reg, X_test=X_test_reg, y_test=y_test_reg)
    evaluator = Evaluator('ann_regressor')
    reg_mse, reg_mae, reg_r2 = evaluator.evaluate(reg_model, X_val=X_val_reg, y_val=y_val_reg, X_test=X_test_reg, y_test=y_test_reg, history=reg_history)
    print(f"PyTorch ANN Regression Results - MSE: {reg_mse:.4f}, MAE: {reg_mae:.4f}, R²: {reg_r2:.4f}, Training Time: {reg_time:.2f} seconds\n")
    
    print("Starting PyTorch ANN Classification Training on CIFAR-100...")
    class_input_size = 3 * 32 * 32
    class_hidden_sizes = [512, 256, 128] 
    class_output_size = 100
    class_lr = 0.001
    class_epochs = 1000
    class_batch_size = 64

    data_loader = DataLoaderModule('classification')
    train_loader, val_loader_class = data_loader.load_data(batch_size=class_batch_size, dataset='CIFAR100')
    class_model = ANN_Classifier(class_input_size, class_hidden_sizes, class_output_size)
    trainer = Trainer('ann_classifier')
    class_model, class_history, class_time = trainer.train(class_model, lr=class_lr, epochs=class_epochs, batch_size=class_batch_size, train_loader=train_loader, val_loader=val_loader_class, dataset='CIFAR100')
    evaluator = Evaluator('ann_classifier')
    class_accuracy, class_cm, class_precision, class_recall, class_f1 = evaluator.evaluate(class_model, val_loader=val_loader_class, history=class_history)
    print(f"PyTorch ANN Classification Results - Accuracy: {class_accuracy:.2f}%, Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1-Score: {class_f1:.4f}, Training Time: {class_time:.2f} seconds\n")
    
    print("Starting Keras CNN Classification Training on CIFAR-10...")
    data_loader = DataLoaderModule('keras_cifar100')
    X_train_keras, y_train_keras, X_val_keras, y_val_keras = data_loader.load_data()
    keras_model = KerasCNN(input_shape=(32, 32, 3), num_classes=100)
    trainer = Trainer('keras_cnn')
    keras_model, keras_history, keras_time = trainer.train(keras_model, X_train=X_train_keras, y_train=y_train_keras, X_val=X_val_keras, y_val=y_val_keras,
                                                              lr=0.0001, epochs=1000, batch_size=64)
    evaluator = Evaluator('keras_cnn')
    keras_accuracy, keras_cm, keras_precision, keras_recall, keras_f1 = evaluator.evaluate(keras_model, X_val=X_val_keras, y_val=y_val_keras, history=keras_history)
    print(f"Keras CNN Classification Results - Accuracy: {keras_accuracy:.2f}%, Precision: {keras_precision:.4f}, Recall: {keras_recall:.4f}, F1-Score: {keras_f1:.4f}, Training Time: {keras_time:.2f} seconds\n")
    
    print("Comparative Table:")
    print(f"{'Model':<25}{'Dataset / Task':<25}{'Key Hyperparams':<50}{'Final Metric':<60}{'Training Time'}")
    print(f"{'PyTorch ANN (Regressor)':<25}{'California Housing':<25}{'LR=0.01, Epoch=50, Batch=64':<50}{f'MSE={reg_mse:.2f}; MAE={reg_mae:.2f}; R²={reg_r2:.2f}':<60}{f'~{reg_time/60:.2f} min'}")
    print(f"{'PyTorch ANN (Classifier)':<25}{'CIFAR-100':<25}{'LR=0.01, Epoch=30, Batch=64':<50}{f'Accuracy={class_accuracy:.2f}%':<60}{f'~{class_time/60:.2f} min'}")
    print(f"{'Keras CNN (Classifier)':<25}{'CIFAR-100':<25}{'LR=0.001, Epoch=50, Batch=64':<50}{f'Accuracy={keras_accuracy:.2f}%':<60}{f'~{keras_time/60:.2f} min'}")

    model_names = ['PyTorch ANN (Regressor)', 'PyTorch ANN (Classifier)', 'Keras CNN (Classifier)']
    metrics = [
        (reg_mse, reg_mae, reg_r2),
        (class_accuracy, class_precision, class_f1),
        (keras_accuracy, keras_precision, keras_f1)
    ]
    training_times = [reg_time/60, class_time/60, keras_time/60]
    
    plot_comparative_metrics(model_names, metrics, training_times)

def plot_comparative_metrics(model_names, metrics, training_times):
    num_models = len(model_names)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].bar(model_names[0], metrics[0][0], label='MSE', color='skyblue')
    axes[0].bar(model_names[0], metrics[0][1], label='MAE', color='coral')
    axes[0].set_title('Regression Metrics')
    axes[0].legend()
    
    axes[1].bar(model_names[1], metrics[1][0], label='Accuracy', color='lightgreen')
    axes[1].bar(model_names[1], metrics[1][1], label='Precision', color='lightcoral')
    axes[1].bar(model_names[1], metrics[1][2], label='F1-Score', color='gold')
    axes[1].set_title('Classification Metrics (PyTorch)')
    axes[1].legend()
    
    axes[2].bar(model_names[2], metrics[2][0], label='Accuracy', color='lightgreen')
    axes[2].bar(model_names[2], metrics[2][1], label='Precision', color='lightcoral')
    axes[2].bar(model_names[2], metrics[2][2], label='F1-Score', color='gold')
    axes[2].set_title('Classification Metrics (Keras)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("comparative_metrics.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, training_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Training Time Comparison')
    plt.ylabel('Training Time (minutes)')
    plt.grid(axis='y')
    plt.savefig("training_time_comparison.png")
    plt.close()

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    main()
