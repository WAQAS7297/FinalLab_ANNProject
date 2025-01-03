import torch
import torch.nn.functional as F
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = torch.randn(in_features, out_features) * torch.sqrt(torch.tensor(2. / in_features))
        self.bias = torch.zeros(out_features)
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)
        
    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        self.grad_weights = self.input.t() @ grad_output
        self.grad_bias = grad_output.sum(0)
        grad_input = grad_output @ self.weights.t()
        return grad_input
    
    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_bias.zero_()

class ANN_Regressor:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(Linear(prev_size, output_size))
 
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer.forward(x))
        x = self.layers[-1].forward(x)
        return x

    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
 
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class ANN_Classifier:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer.forward(x))
        x = self.layers[-1].forward(x)
        x = F.softmax(x, dim=1)
        return x

    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class KerasCNN:
    def __init__(self, input_shape=(32, 32, 3), num_classes=100):
        self.model = self._create_model(input_shape, num_classes)

    def _create_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model
