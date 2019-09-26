from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    # get our data
    iris = load_iris()
    dataX = iris['data']
    dataY = iris['target']

    print(dataX)
    #print(dataY)
    #print(iris.target_names)