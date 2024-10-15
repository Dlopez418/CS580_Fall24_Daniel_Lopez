import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'linear_regression_data.csv'

data = pd.read_csv(file_path, header=None)

column_1 = data[0].tolist()  
column_2 = data[1].tolist()  

b1 = 0.0
b0 = 0.0

X_Y = 0.0
X=0.0
Y=0.0
X2=0.0
n = len(column_1)

for num in column_1: 
    X+=num
    X2+=(num*num)

for mun in column_2:
    Y+=mun
    X_Y += mun * column_1[column_2.index(mun)]
    

b1 = ((n*X_Y) - (X*Y))/((n*X2)-(X*X))
b0 = (Y-(b1*X))/n

print("y=",b0,"+",b1,"X")

x_values = column_1
y_values = column_2


x_range = np.linspace(min(x_values),max(x_values),100)

y_range = [b0 + b1 * x for x in x_range]


plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', label='Data Points', s=100) 
plt.plot(x_range, y_range, color='red', label='Regression Line')  
plt.title('Linear Regression')
plt.xlabel('X values (Column 1)')
plt.ylabel('Y values (Column 2)')
plt.legend()
plt.grid(True)
plt.xlim(min(x_values) - 1, max(x_values) + 1)
plt.ylim(min(y_values) - 1, max(y_values) + 1)

plt.show()
