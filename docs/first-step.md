# Resnet 50 Example

## Basic Concepts



## Recommended Folder Structure
It's recommended to construct your project in the following form, which the preprocess folder is keeping all dataset loaders, and the models folder keeps all the models. However, you can orgnize whatever you want.
```
--ProjectFolder
---data
----dataset1
----dataset2
----...
---preprocess
    cifia10_loader.py
    dataset2_loader.py
    ...
---models
    resnet50.py
    Lnet.py
    ...
main.py
```

## Prepare Datasets
The start of a deep learning experiment is always either be preparing datasets or constructing models. In this example, we first prepare dataset. 

In the cifia10_loder.py

```python
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from LightningLite.DataModule import DataModule
from sklearn.model_selection import KFold
import numpy as np


class DatasetLoader(DataModule):
    def __init__(self, random_state, dataset: str = 'pima-indians-diabetes'):
        self.dataset = dataset
        kf = KFold(n_splits=10)
        self.random_state = random_state
        self.X, self.y, self.num_features, self.num_classes = self.load_pima()
        self.kf = kf.split(self.X, self.y)

    def train_loader(self, test_size=0.2):
        train, test = next(self.kf)
        X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]
        return X_train, X_test, y_train, y_test, self.num_features, self.num_classes

    def load_pima(self, path='./data/pima-indians-diabetes.csv'):
        num_features = 8
        num_classes = 2
        dataset = loadtxt(path, delimiter=",")
        shuffle = np.random.RandomState(seed=self.random_state)
        shuffle.shuffle(dataset)
        X = dataset[:, :num_features]
        y = dataset[:, -1]
        return X, y, num_features, num_classes
```

The DataModule of LightningLite is bascially a 'empty' class. It only give regulations on your codes.

The dataset passed should be iteratable. And each epoch the trainer will trasel through it and pass each item into LiteModule as training batch data.


## Construct Model


```python
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from LightningLite.DataModule import DataModule
from sklearn.model_selection import KFold
import numpy as np


class DatasetLoader(DataModule):
    def __init__(self, random_state, dataset: str = 'pima-indians-diabetes'):
        self.dataset = dataset
        kf = KFold(n_splits=10)
        self.random_state = random_state
        self.X, self.y, self.num_features, self.num_classes = self.load_pima()
        self.kf = kf.split(self.X, self.y)

    def train_loader(self, test_size=0.2):
        train, test = next(self.kf)
        X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]
        return X_train, X_test, y_train, y_test, self.num_features, self.num_classes

    def load_pima(self, path='./data/pima-indians-diabetes.csv'):
        num_features = 8
        num_classes = 2
        dataset = loadtxt(path, delimiter=",")
        shuffle = np.random.RandomState(seed=self.random_state)
        shuffle.shuffle(dataset)
        X = dataset[:, :num_features]
        y = dataset[:, -1]
        return X, y, num_features, num_classes
```


## Training and Validation


## Test


## Predict



## Draw charts from logs