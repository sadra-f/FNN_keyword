# FeedForward Neural Network from scratch

In order to gain a better understanding of Neural networks and Backpropagation, I attempted to implement an FNN from scratch. 

I implemented this Feedforward Nerual Network from scratch, so minimal libraries are used. numpy for the FNN and pandas for some dataset manipulation.
I previously Generated a structured numerical keyword prediction dataset which was used in [this](https://github.com/sadra-f/LRKeywordExtraction) project (view for more details on dataset) where I Implemented Logistic Regression from scratch and used the mentioned dataset as an experimental keyword extraction method.

## Sample Code
Sample to run train/test the FNN with the MNIST dataset available in repo.
``` python
from MLModel.FNN import FeedForwardNeuralNetwork as FNN
from IO.Read import read_csv_dataset
from Evaluation.classification_eval import evaluate_classification

train = read_csv_dataset("./dataset/mnist_train.csv")
train_Y = train['label'].to_numpy()
train_X = train.drop('label', axis=1).to_numpy()

fnn = FNN()
fnn.build(784, [128], 10)
fnn.train(train_X, train_Y, 3)

test = read_csv_dataset("./dataset/mnist_test.csv")
test_Y = test['label'].to_numpy()
test_X = test.drop('label', axis=1).to_numpy()

predictions = fnn.predict_all(test_X)
evaluate_classification(test_Y, predictions)
```
## Results
The models performance on the Keyword dataset is provided in the table below which shows an improvement in comparison to my [Logistic Regression model performance](https://github.com/sadra-f/LRKeywordExtraction?tab=readme-ov-file#performance-evaluation) on the same dataset:
| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.718    | 0.762     | 0.639  | 0.695    |

The models performance on the MNIST handwritten digits dataset is also provided below.
|Accuracy|
|--------|
|0.906|

| digit | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Precision | 0.926 | 0.980 | 0.920 | 0.878 | 0.877 | 0.896 | 0.925 | 0.915 | 0.839 | 0.900 |

| digit | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Recall | 0.977 | 0.966 | 0.858 | 0.904 | 0.933 | 0.837 | 0.926 | 0.908 | 0.885 | 0.856 |


| digit | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| F1 | 0.951 | 0.973 | 0.888 | 0.891 | 0.904 | 0.866 | 0.926 | 0.912 | 0.861 | 0.878