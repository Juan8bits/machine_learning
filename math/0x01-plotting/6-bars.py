#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruits = ['apples', 'bananas', 'oranges', 'peaches']
customers = ['Farrah', 'Fred', 'Felicia']
plt.title('Number of Fruit per Person')
plt.bar(customers, fruit[0], label=fruits[0], color='red', width=0.5)
plt.bar(customers, fruit[1], bottom=fruit[0], label=fruits[1],
        color='yellow', width=0.5)
plt.bar(customers, fruit[2], bottom=fruit[1] + fruit[0],
        label=fruits[2], color='#ff8000', width=0.5)
plt.bar(customers, fruit[3], bottom=fruit[2] + fruit[1] + fruit[0],
        label=fruits[3], color='#ffe5b4', width=0.5)
plt.ylim([0, 80])
plt.yticks(range(0, 90, 10))
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.show()
