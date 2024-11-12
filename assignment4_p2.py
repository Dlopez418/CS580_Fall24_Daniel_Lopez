import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'data.csv'
data = pd.read_csv(file_path, header=None)

column_1 = data[0].to_numpy()  
column_2 = data[1].to_numpy()  
column_3 = data[2].to_numpy()


w_1 = np.random.uniform(-1.0,1.0)
w_2 = np.random.uniform(-1.0,1.0)
b = np.random.uniform(-1.0,1.0)
lr = 0.01
threshold = 0.01
log_losses = []

plt.figure(figsize=(1.0,1.0))

plt.scatter(column_1[column_3 == 0], column_2[column_3 == 0], color='blue', label='Class 0')
plt.scatter(column_1[column_3 == 1], column_2[column_3 == 1], color='red', label='Class 1')

x_vals = np.linspace(min(column_1), max(column_1), 100)
initial_y_vals = - (w_1 * x_vals + b) / w_2
plt.plot(x_vals, initial_y_vals, color='red', label='Initial Separation Line')



for epoch in range(100):
    for index in range(len(column_1)):
        linear = (column_1[index]*w_1)+(column_2[index]*w_2)+b
        y_hat  = 1/(1 + np.exp(-linear))
        error = (column_3[index] - y_hat)
        b = b + (lr * error)
        w_1 = w_1 + (lr * error * column_1[index])
        w_2 = w_2 + (lr * error * column_2[index])

    y_vals = - (w_1 * x_vals + b) / w_2
    plt.plot(x_vals, y_vals, color='green', linestyle='--', alpha=0.5)
    


    if np.abs(error) < threshold:
        break

final_y_vals = - (w_1 * x_vals + b) / w_2
plt.plot(x_vals, final_y_vals, color='black', label='Final Separation Line')

plt.xlim(0,1)
plt.ylim(0,1)


plt.title('Perceptron using Gradient Descent')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()
plt.savefig('Assignment4_p1')