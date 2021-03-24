import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
np.random.seed(1)
b = np.random.choice(a, 4, replace=False)
print(b)
