from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # step #1, get some data
    data = load_iris()
    dataX = data['data']
    dataY = data['target']

    # step #2, split data into training & testing
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3)
        
    # step #3, create a learner
    learner = KNeighborsClassifier()

    # step #4, fit the data
    learner.fit(trainX, trainY)

    # step #5, have learner predict labels for our test data
    predictY = learner.predict(testX)

    # step #6, get metrics
    accuracy = accuracy_score(testY, predictY)
    print(accuracy)