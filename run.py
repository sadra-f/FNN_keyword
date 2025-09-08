from MLModel.FNN_retry import FeedForwardNeuralNetwork as FNN
from IO.Read import read_csv_dataset
from Evaluation.classification_eval import binary_classification_metrics

import numpy as np
import pandas as pd

dataset = read_csv_dataset("./dataset/numerical_dataset.csv")
dataset['keyword'] = dataset['keyword'].replace(False, 0).replace(True, 1)

split = np.random.rand(len(dataset)) < 0.7

train = dataset[split]
test = dataset[~split]

X_train = train.drop(['word', 'keyword'], axis=1).to_numpy()
Y_train = train['keyword'].to_numpy()

X_test = test.drop(['word', 'keyword'], axis=1).to_numpy()
Y_test = test['keyword'].to_numpy()

res = None
nn = FNN()
nn.build(48, [128], 2)
nn.train(X_train, Y_train, epoch=22)

predictions = []
for i in range(len(X_test)):
    predictions.append(nn.predict(X_test[i]))


res.append(binary_classification_metrics(Y_test, predictions))
#uncomment to save loss history for each iteration of each epoch as txt file
# np.savetxt("loss.csv", np.array(nn.loss_hist))
print(res)