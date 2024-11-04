```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)

number_of_points = 50
x_scale = 10
gradient = 2
y_intercept = -5

x = x_scale * rng.rand(number_of_points)
y = gradient * x + y_intercept + rng.normal(size=number_of_points)

data = pd.DataFrame({"x": x, "y": y})

data.to_csv("linear.csv", index=False)
```
