import numpy as np

def l21shrink(epsilon, x):
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        elif norm[i] < -epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] + epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output