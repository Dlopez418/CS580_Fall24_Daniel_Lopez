import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'data.csv'
data = pd.read_csv(file_path, header=None)

column_1 = data[0].to_numpy()  
column_2 = data[1].to_numpy()  
column_3 = data[2].to_numpy()

w_1 = 1.0
w_2 = -1.0
b = 0.5
lr = 1

x_values = np.linspace(min(column_1), max(column_1), 100)
y_values = -((w_1 * x_values + b) / w_2)

plt.figure(figsize=(0.0, 1.0))

# Plot points for each class
plt.scatter(column_1[column_3 == 0], column_2[column_3 == 0], color='blue', label='Class 0', alpha=0.5)
plt.scatter(column_1[column_3 == 1], column_2[column_3 == 1], color='red', label='Class 1', alpha=0.5)

plt.plot(x_values, y_values, color='red', label='Initial Decision Boundary')

for epoch in range(100):
    for num in range(len(column_1)):
        linear = (column_1[num]*w_1)+(column_2[num]*w_2)+b
        if linear < 0 and column_3[num]==1:
            b = b+lr
            w_1 = w_1 + (lr * column_1[num])
            w_2 = w_2 + (lr*column_2[num])
       
        elif linear >= 0 and column_3[num]==0:  
            b = b -lr
            w_1 = w_1 - (lr * column_1[num])
            w_2 = w_2 - (lr*column_2[num])
    y_current = -((w_1 * x_values + b) / w_2)
    plt.plot(x_values, y_current, color='green', linestyle='--', alpha=0.5)

y_final = -((w_1 * x_values + b) / w_2)

plt.plot(x_values, y_final, color='black', label='Final Decision Boundary')

plt.xlim(0,1)
plt.ylim(0,1)


plt.title('Solution Boundary')
plt.legend()
plt.grid()
plt.show()



