from data_loader import DataLoaderModule
from models import ANN_Regressor, ANN_Classifier, KerasCNN
from trainer import Trainer
from evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("Starting PyTorch ANN Regression Training...")
    reg_input_size = 8
    reg_hidden_sizes = [64, 64]
    reg_output_size = 1
    reg_lr = 0.01
    reg_epochs = 50
    reg_batch_size = 64

    data_loader = DataLoaderModule('regression')
    X_train, y_train, X_val_reg, y_val_reg = data_loader.load_data()
    reg_model = ANN_Regressor(reg_input_size, reg_hidden_sizes, reg_output_size)
    trainer = Trainer('ann_regressor')
    reg_model, reg_time = trainer.train(reg_model, lr=reg_lr, epochs=reg_epochs, batch_size=reg_batch_size, X_train=X_train, y_train=y_train, X_val=X_val_reg, y_val=y_val_reg)
    evaluator = Evaluator('ann_regressor')
    reg_mse, reg_mae, reg_r2 = evaluator.evaluate(reg_model, X_val=X_val_reg, y_val=y_val_reg)
    print(f"PyTorch ANN Regression Results - MSE: {reg_mse:.4f}, MAE: {reg_mae:.4f}, R²: {reg_r2:.4f}, Training Time: {reg_time:.2f} seconds\n")
    
    print("Starting Keras CNN Classification Training on CIFAR-100...")
    data_loader = DataLoaderModule('keras_cifar100')
    X_train_keras, y_train_keras, X_val_keras, y_val_keras = data_loader.load_data()
    keras_model = KerasCNN(input_shape=(32, 32, 3), num_classes=100)
    trainer = Trainer('keras_cnn')
    keras_model, keras_history, keras_time = trainer.train(keras_model, X_train=X_train_keras, y_train=y_train_keras, X_val=X_val_keras, y_val=y_val_keras,
                                                              lr=0.001, epochs=50, batch_size=64)
    evaluator = Evaluator('keras_cnn')
    keras_accuracy, keras_cm, keras_precision, keras_recall, keras_f1 = evaluator.evaluate(keras_model, X_val=X_val_keras, y_val=y_val_keras, history=keras_history)
    print(f"Keras CNN Classification Results - Accuracy: {keras_accuracy:.2f}%, Precision: {keras_precision:.4f}, Recall: {keras_recall:.4f}, F1-Score: {keras_f1:.4f}, Training Time: {keras_time:.2f} seconds\n")


if __name__ == "__main__":
    main()