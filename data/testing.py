import pandas as pd

data = pd.read_csv("https://bristol-training.github.io/applied-data-analysis-in-python/data/linear.csv")

X = data[["x"]]
y = data["y"]

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

import seaborn as sns

# Label the original DataFrame with the test/train split
# This is just used for plotting purposes
data.loc[train_X.index, "train/test"] = "train"
data.loc[test_X.index, "train/test"] = "test"

sns.relplot(data=data, x="x", y="y", hue="train/test")

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(train_X, train_y)

model.score(test_X, test_y)
