# Asking questions of your data

In the previous course we 


```python
import pandas as pd
from pandas import Series, DataFrame

data = {'city': ['Paris', 'Paris', 'Paris', 'Paris',
                 'London', 'London', 'London', 'London',
                 'Rome', 'Rome', 'Rome', 'Rome'],
        'year': [2001, 2008, 2009, 2010,
                 2001, 2006, 2011, 2015,
                 2001, 2006, 2009, 2012],
        'pop': [2.148, 2.211, 2.234, 2.244,
                7.322, 7.657, 8.174, 8.615,
                2.547, 2.627, 2.734, 2.627]}
df = DataFrame(data)
df
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
      <th>city</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paris</td>
      <td>2001</td>
      <td>2.148</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>2008</td>
      <td>2.211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>2009</td>
      <td>2.234</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Paris</td>
      <td>2010</td>
      <td>2.244</td>
    </tr>
    <tr>
      <th>4</th>
      <td>London</td>
      <td>2001</td>
      <td>7.322</td>
    </tr>
    <tr>
      <th>5</th>
      <td>London</td>
      <td>2006</td>
      <td>7.657</td>
    </tr>
    <tr>
      <th>6</th>
      <td>London</td>
      <td>2011</td>
      <td>8.174</td>
    </tr>
    <tr>
      <th>7</th>
      <td>London</td>
      <td>2015</td>
      <td>8.615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rome</td>
      <td>2001</td>
      <td>2.547</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rome</td>
      <td>2006</td>
      <td>2.627</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rome</td>
      <td>2009</td>
      <td>2.734</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rome</td>
      <td>2012</td>
      <td>2.627</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.query("year == 2001")
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
      <th>city</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paris</td>
      <td>2001</td>
      <td>2.148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>London</td>
      <td>2001</td>
      <td>7.322</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rome</td>
      <td>2001</td>
      <td>2.547</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df["year"] == 2001]
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
      <th>city</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paris</td>
      <td>2001</td>
      <td>2.148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>London</td>
      <td>2001</td>
      <td>7.322</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rome</td>
      <td>2001</td>
      <td>2.547</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.query("city == 'Paris'")["pop"].max()
```




    2.244



[<font size="5">Previous</font>](Introduction.qmd)<font size="5"> | </font>[<font size="5">Next</font>](Fitting.qmd)
