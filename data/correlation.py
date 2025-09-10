from sklearn.datasets import fetch_california_housing

housing, target = fetch_california_housing(as_frame=True, return_X_y=True)



housing.head()


housing.corr()



corr = housing.corr()
corr["MedInc"]["AveRooms"]


import seaborn as sns

sns.heatmap(corr, vmin=-1.0, vmax=1.0, square=True, cmap="RdBu")


from pandas.plotting import scatter_matrix

a = scatter_matrix(housing, figsize=(16, 16))


