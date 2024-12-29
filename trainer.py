import time
import torch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


class Trainer:
    def __init__(self, model_type):
        self.model_type = model_type

    def sgd(self, model, lr):
        for layer in model.layers:
            layer.weights -= lr * layer.grad_weights
            layer.bias -= lr * layer.grad_bias

    def train(self, model, **kwargs):
        if self.model_type == 'ann_regressor':
            return self._train_regression(model, **kwargs)
        elif self.model_type == 'ann_classifier':
            return self._train_classification(model, **kwargs)
        elif self.model_type == 'keras_cnn':
            return self._train_keras_cnn(model, **kwargs)
        else:
            raise ValueError("Unsupported model type.")

    def _train_regression(self, model, lr, epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test):
        loss_fn = torch.nn.MSELoss()
        start_time = time.time()
        history = {'train_losses': [], 'val_losses': []}
        for epoch in range(epochs):
            permutation = torch.randperm(X_train.size()[0])
            epoch_loss = 0
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                pred = model.forward(batch_x)
                loss = loss_fn(pred, batch_y)
                epoch_loss += loss.item()
                grad_loss = (pred - batch_y) * 2 / batch_y.size(0)
                model.backward(grad_loss)
                self.sgd(model, lr)
                model.zero_grad()
            val_pred = model.forward(X_val)
            val_loss = loss_fn(val_pred, y_val).item()
            history['train_losses'].append(epoch_loss)
            history['val_losses'].append(val_loss)
            print(f"Reg Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        end_time = time.time()
        training_time = end_time - start_time
        return model, history, training_time

    def _train_classification(self, model, lr, epochs, batch_size, train_loader, val_loader, dataset='CIFAR10'):
        loss_fn = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        history = {'train_losses': [], 'val_losses': []}
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                pred = model.forward(batch_x)
                loss = loss_fn(pred, batch_y)
                epoch_loss += loss.item()
                batch_size_current = batch_y.size(0)
                grad_loss = pred.clone()
                grad_loss[range(batch_size_current), batch_y] -= 1
                grad_loss /= batch_size_current
                model.backward(grad_loss)
                self.sgd(model, lr)
                model.zero_grad()
            val_loss = 0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.view(val_x.size(0), -1)
                    pred = model.forward(val_x)
                    loss = loss_fn(pred, val_y)
                    val_loss += loss.item()
            history['train_losses'].append(epoch_loss)
            history['val_losses'].append(val_loss)
            print(f"Class Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        end_time = time.time()
        training_time = end_time - start_time
        return model, history, training_time

    def _train_keras_cnn(self, model, X_train, y_train, X_val, y_val, lr=0.001, epochs=20, batch_size=64):
        model.model.compile(optimizer=Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        start_time = time.time()
        history = model.model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stop],
                            verbose=2)
        end_time = time.time()
        training_time = end_time - start_time
        return model, history, training_time