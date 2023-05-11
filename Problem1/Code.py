import numpy as np
def toActualIndex(l, N):
    return N + l

def toSpatialDomian(u, x, N):
    mul = np.linspace(-N, N, 2*N +1, endpoint=True)
    xComp = 2j * np.pi * x * mul
    u_spatial = u*np.exp( xComp)

    return np.sum(u_spatial)

def getA(N):
    A = np.zeros((2*N + 1, 2*N +1))
    for i in range(-N, N+1):
        k = i - 1
        if k >= -N:
            A[toActualIndex(i, N), toActualIndex(k, N)] = np.pi * i
        k = i + 1
        if k <= N:
            A[toActualIndex(i, N), toActualIndex(k, N)] = -np.pi * i
    return A

def getD2(N):
    D2 = []
    for i in range(-N, N+1):
        D2.append(-4 * (np.pi**2) * i **2)
    D2 = np.diag(D2)
    return D2
if __name__ == '__main__':
    N = 10

    u = np.zeros(N*2 + 1, dtype=np.complex128)

    u0 = np.copy(u)
    u0[toActualIndex(1, N)] = 1 / 2j
    u0[toActualIndex(-1, N)] = -1 / 2j

    print(u0)

    print(toSpatialDomian(u0, 0.125, N))

    A = getA(N)
    # print(A)

    D2 = getD2(N)
    # print(D2)

    H = -A + 0.5 * D2

    u, s, vh =np.linalg.svd(H)
    print(s)


