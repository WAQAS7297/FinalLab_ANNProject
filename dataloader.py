import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import cifar100


class DataLoaderModule:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

    def load_data(self, **kwargs):
        if self.dataset_type == 'regression':
            return self._load_regression_data(**kwargs)
        elif self.dataset_type == 'classification':
            return self._load_classification_data(**kwargs)
        elif self.dataset_type == 'keras_cifar100':
            return self._load_keras_cifar100_data()
        else:
            raise ValueError("Unsupported dataset type.")

    def _load_regression_data(self, test_size=0.2):
        data = fetch_california_housing()
        X = data.data
        y = data.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
        return X_train, y_train, X_val, y_val

    def _load_keras_cifar100_data(self):
        (X_train, y_train), (X_val, y_val) = cifar100.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        return X_train, y_train, X_val, y_val