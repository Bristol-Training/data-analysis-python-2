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

To find the index of the pixel with the largest blue value, we use `idxmax()` on the `Blue` column.


```python
bluest_index = pixels["Blue"].idxmax()
```

We use the width, `w`, of the original image as the denominator for a division and a [modulo](https://en.wikipedia.org/wiki/Modulo_operation).


```python
x = bluest_index % w
y = bluest_index // w
x, y
```




    (537, 150)


