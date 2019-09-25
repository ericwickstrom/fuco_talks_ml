import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

if __name__ == '__main__':
    # step #1, get some data
    data = datasets.load_iris()
    dataX = data['data']
    dataY = data['target']

    # step #2, split data into training & testing
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3)
    
    # step #3, create a learner
    learner = KNeighborsClassifier()

    # step #4, create tuple of parameters to try
    param_grid = {'weights':['distance', 'uniform'], 'metric':['chebyshev', 'euclidean','manhattan'], 'n_neighbors': np.arange(1,51,1)}
    
    # step #5, use grid search to find best parameters
    grid = GridSearchCV(learner, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(trainX, trainY)
    score = grid.score(testX, testY)

    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(score))
    print("Params: {}".format(grid.best_params_))