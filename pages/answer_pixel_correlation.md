```python
import numpy as np
from pandas import DataFrame, Series
from skimage import io

photo = io.imread("https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Swallow-tailed_bee-eater_%28Merops_hirundineus_chrysolaimus%29.jpg/768px-Swallow-tailed_bee-eater_%28Merops_hirundineus_chrysolaimus%29.jpg")

photo = np.array(photo, dtype=np.float64) / 255  # Scale values
w, h, d = original_shape = tuple(photo.shape)  # Get the current shape
image_array = np.reshape(photo, (w * h, d))  # Reshape to to 2D

pixels = DataFrame(image_array, columns=["Red", "Green", "Blue"])
```


```python
pixels.corr()
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
      <th>Red</th>
      <th>Green</th>
      <th>Blue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Red</th>
      <td>1.000000</td>
      <td>0.85383</td>
      <td>0.749801</td>
    </tr>
    <tr>
      <th>Green</th>
      <td>0.853830</td>
      <td>1.00000</td>
      <td>0.781750</td>
    </tr>
    <tr>
      <th>Blue</th>
      <td>0.749801</td>
      <td>0.78175</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


