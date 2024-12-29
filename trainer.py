import time
import torch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class Trainer:
    def __init__(self, model_type):
        self.model_type = model_type

    def train(self, model, **kwargs):
        if self.model_type == 'ann_regressor':
            return self._train_regression(model, **kwargs)
        elif self.model_type == 'ann_classifier':
            return self._train_classification(model, **kwargs)
        elif self.model_type == 'keras_cnn':
            return self._train_keras_cnn(model, **kwargs)
        else:
            raise ValueError("Unsupported model type.")

    def _train_regression(self, model, lr, epochs, batch_size, X_train, y_train, X_val, y_val):
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
        return model, training_time

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
