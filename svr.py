import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt

data = pd.read_csv('./Position_Salaries.csv')

x = data.iloc[:, 1:2].values
y = data.iloc[:, 2:].values

sc_x = preprocessing.StandardScaler()
x = sc_x.fit_transform(x)
sc_y = preprocessing.StandardScaler()
y = sc_y.fit_transform(y)

svr_regressor = svm.SVR(kernel = 'rbf')
svr_regressor.fit(x, y)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, svr_regressor.predict(x_grid), color = 'blue')
plt.title('SVR plot')
plt.xlabel('Experience level')
plt.ylabel('Salary')
plt.show()

print(sc_y.inverse_transform(svr_regressor.predict(sc_x.fit_transform(np.array([[6.5]])))))