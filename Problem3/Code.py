import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
from matplotlib import pyplot as plt
from scipy.special import legendre

if __name__ == '__main__':
    N = 10

    legendres = []

    for i in range(N):
        legendres.append(legendre(i))
        print(legendres[-1])

