import test_lib as Tester
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams["figure.figsize"] = (11,11)

#Demo - part 1
x = np.linspace(0,1176,168)
data_names = ['date','averageprice','totalvalue','4046','40225','4770','totalbags','smallbags','largebags','Xlargebags','type','year','region']
data = pd.read_csv('avocado.csv',delimiter = ',', names = data_names,nrows = 168)

y = data['totalvalue'].values
plt.scatter(x, y, s=100)
#Tester.fitLineAndPlot(x, y, plt, order=5)
Tester.fitLineAndPlot(x, y, plt, order=1)
Tester.fitLineAndPlot(x, y, plt, order=2)
Tester.fitLineAndPlot(x, y, plt, order=3)
Tester.fitLineAndPlot(x, y, plt, order=4)
#Tester.fitLineAndPlot(x, y, plt, order=5)
plt.rcParams["figure.figsize"] = (11,11)
a1 = 94924.81716
b = -267.77003
c = 1.072081
d = -0.00139566
e = 5.973624813824965e-07
a = np.linspace(0,1350,200)
y1 = [a1 + b * a[i] + c * a[i] ** 2 + d * a[i] ** 3 + e * a[i] ** 4 for i in range(0,len(a))]
plt.plot(a,y1)

plt.show()
