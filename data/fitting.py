import pandas as pd

data = pd.read_csv("https://bristol-training.github.io/applied-data-analysis-in-python/data/linear.csv")
data.head()

import seaborn as sns

sns.relplot(data=data, x="x", y="y")

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model

X = data[["x"]]
y = data["y"]

model.fit(X, y)

pred = pd.DataFrame({"x": [0, 10]})  # Make a new DataFrame containing the X values
pred["y"] = model.predict(pred)  # Make a prediction and add that data into the table
pred

import seaborn as sns
sns.relplot(data=data, x="x", y="y")
sns.lineplot(data=pred, x="x", y="y", c="red", linestyle=":")
