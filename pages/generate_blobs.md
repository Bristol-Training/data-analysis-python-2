```python
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=4, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X, columns=["x1", "x2"])

sns.scatterplot(data=X, x="x1", y="x2", hue=y, palette="Dark2")
```




    <Axes: xlabel='x1', ylabel='x2'>




    
![](../img/generate_blobs_0_1.png)
    



```python
data = X.copy()
data["y"] = y
```


```python
data.to_csv("blobs.csv", index=False)
```


```python
check = pd.read_csv("blobs.csv")
pd.testing.assert_frame_equal(data, check)
```
