```python
import numpy as np
from pandas import Series, DataFrame

a = np.arange(100)
b = np.arange(100) * -2
df = DataFrame({"a": a, "b": b})

df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


