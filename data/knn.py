import pandas as pd

data = pd.read_csv("https://bristol-training.github.io/applied-data-analysis-in-python/data/moons.csv")

data.head()


import seaborn as sns

sns.scatterplot(data=data, x="x1", y="x2", hue="y", palette="Dark2")

X = data[["x1", "x2"]]
y = data["y"]


from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)



model.score(test_X, test_y)


from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(model, X, cmap="PRGn")
sns.scatterplot(data=X, x="x1", y="x2", hue=y, palette="Dark2")

