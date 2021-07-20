#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

customers = ["Farrah", "Fred", "Felicia"]
fruit_names = ["apples", "bananas", "oranges", "peaches"]
fruit_colors = ["red", "yellow", "orange", "#ffe5b4"]
number_cust = np.arange(len(customers))

plt.xticks(number_cust, customers)
plt.ylim([0, 80])
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")

for i in range(len(fruit_names)):
    plt.bar(number_cust, fruit[i, :], label=fruit_names[i], color=fruit_colors[i],
            width=0.5, bottom=np.apply_along_axis(sum, 0, fruit[:i, :]))

plt.legend()
plt.show()
