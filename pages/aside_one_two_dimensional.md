scitkit-learn requires the `X` parameter of the `fit()` function to be two-dimensional and the `y` parameter to be one-dimensional.

`X` must be two-dimensional, even if there is only one feature (column) present in your data. This can sometimes be a bit confusing as to humans there's little difference between a table with one column and a simple list of values. Computers, however are very explicit about this difference and so we need to make sure we're doing the right thing.

First, let's grab the data we were working with:


```python
from pandas import read_csv

data = read_csv("https://bristol-training.github.io/applied_data_analysis_in_python/linear.csv")
```

## 2D `DataFrame`s

If we look at it, we see it's a pandas `DataFrame` which is always inherently two-dimensional:


```python
data.head()
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.745401</td>
      <td>3.229269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.507143</td>
      <td>14.185654</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.319939</td>
      <td>9.524231</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.986585</td>
      <td>6.672066</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.560186</td>
      <td>-3.358149</td>
    </tr>
  </tbody>
</table>
</div>



To get a more specific idea of the shape of the data structure, we can use the `shape` attribute:


```python
data.shape
```




    (50, 2)



This tell us that it's a $(50 \times 2)$ structure so is two dimensional.

To be explicit, we can also query its dimensionality directly with `ndim`:


```python
data.ndim
```




    2



## 1D `Series`

If we ask a `DataFrame` for one of its columns, it returns it to us as a pandas `Series`. These objects are always one-dimensional (ignoring the potential for multi-indexes):


```python
data["x"].head()
```




    0    3.745401
    1    9.507143
    2    7.319939
    3    5.986585
    4    1.560186
    Name: x, dtype: float64




```python
type(data["x"])
```




    pandas.core.series.Series




```python
data["x"].shape
```




    (50,)



Note that the `shape` is `(50,)`. This might look like it could have multiple values but this is just how Python represents a tuple with one value. To check the dimensionality explicitly, we can peek at `ndim` again:


```python
data["x"].ndim
```




    1



## 2D subsets of `DataFrame`s

If we want to ask a `DataFrame` for a subset of its columns, it will return the answer to us as a another `DataFrame` as this is the only way to represent data with multiple columns.

We can ask for multiple columns by passing a list of column names to the `DataFrame` indexing operator.

Pay attention here as the *outer pair* of square brackets are denoting the indexing operator being called while the *inner pair* denotes the list being created.


```python
data[["x", "y"]].head()
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.745401</td>
      <td>3.229269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.507143</td>
      <td>14.185654</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.319939</td>
      <td>9.524231</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.986585</td>
      <td>6.672066</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.560186</td>
      <td>-3.358149</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[["x", "y"]].shape
```




    (50, 2)



We can see here that when we asked the `DataFrame` for multiple columns by passing a list of column names it returns a two-dimensional object.

If we want to extract just one column but still maintain the dimensionality, we can pass a list with only one column name:


```python
data[["x"]].head()
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
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.745401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.507143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.319939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.986585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.560186</td>
    </tr>
  </tbody>
</table>
</div>



If we check the shape and dimensionality of this, we see that it is a $(50 \times 1)$ structure with two dimensions:


```python
data[["x"]].shape
```




    (50, 1)




```python
data[["x"]].ndim
```




    2



## Final comparison

Finally, to reiterate, the difference between


```python
data["x"].head()
```




    0    3.745401
    1    9.507143
    2    7.319939
    3    5.986585
    4    1.560186
    Name: x, dtype: float64



and


```python
data[["x"]].head()
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
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.745401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.507143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.319939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.986585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.560186</td>
    </tr>
  </tbody>
</table>
</div>



is not really in the data itself, but in the mathematical structure. One is a vector and and the other is a matrix. One is one-dimensional and the other is two-dimensional. 


```python
data["x"].ndim
```




    1




```python
data[["x"]].ndim
```




    2


