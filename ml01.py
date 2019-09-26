from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    # get our data
    iris = load_iris()
    dataX = iris['data']
    dataY = iris['target']

    # add the feature names
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # for reference, our labels
    classes = ('setosa', 'versicolour', 'verginica')

    df = pd.DataFrame(data=dataX, columns=features)
    df['labels'] = dataY
    df['labels'] = df['labels'].map({0: 'setosa', 1: 'versicolour', 2: 'verginica'})

    sns.set()
    g = sns.scatterplot(x='sepal length', y='sepal width', data=df, hue='labels', palette=['green', 'red', 'blue'])
    plt.show(g)

    g = sns.scatterplot(x='petal length', y='petal width', data=df, hue='labels', palette=['green', 'red', 'blue'])
    plt.show(g)