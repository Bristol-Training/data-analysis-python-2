import pandas as pd

data = pd.read_csv("https://bristol-training.github.io/applied-data-analysis-in-python/data/blobs.csv")
X = data[["x1", "x2"]]
y = data["y"]

import seaborn as sns

sns.scatterplot(data=data, x="x1", y="x2", hue="y", palette="Dark2")


from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y)


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

hyperparameters = {
    "n_neighbors" : range(1, 175),
}
model = GridSearchCV(KNeighborsClassifier(), hyperparameters)
model.fit(train_X, train_y)



cv_results = pd.DataFrame(model.cv_results_)
cv_results.plot.scatter("param_n_neighbors", "mean_test_score", yerr="std_test_score", figsize=(10,8))



from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(model, X, cmap="Pastel2")
sns.scatterplot(data=X, x="x1", y="x2", hue=y, palette="Dark2")


new_X = pd.DataFrame({
    "x1": [0, -10, 5, -5],
    "x2": [10, 5, 0, -10],
})

model.predict(new_X)


model.score(test_X, test_y)


