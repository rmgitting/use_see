import numpy as np
from tasea.utils import Utils


def calculate_distances(t, s, key):
    if key == 'brute':
        return euclidean_distance_unequal_lengths(t, s)
    elif key == 'mass':
        return mass(t, s)
    elif key == 'dtw':
        return dtw_distance(t, s)


def euclidean_distance(t1, t2):
    return np.sqrt(sum((t1 - t2) ** 2))


def euclidean_distance_unequal_lengths(t, s):
    distances = np.array([euclidean_distance(np.array(s1), s) for s1 in Utils.sliding_window(t, len(s))])
    return distances


def dtw_distance(s1, s2):
    dtw = {}

    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


def dot_products(s, t):
    m, n = len(s), len(t)
    t_a = np.concatenate([t, np.zeros(n)])
    s_r = s[::-1]
    s_ra = np.concatenate([s_r, np.zeros(2 * n - m)])
    s_raf = np.fft.fft(s_ra)
    t_af = np.fft.fft(t_a)
    st = np.fft.ifft(s_raf * t_af)
    return st


def mass(x, y):
    n = len(x)
    # y = (y-np.mean(y))/ np.std(y)
    m = len(y)
    x = np.concatenate([x, np.zeros(n)])
    y = y[::-1]
    y = np.concatenate([y, np.zeros(2 * n - m)])
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z)
    z = z.real

    sumy = np.sum(y)
    sumy2 = np.sum(y ** 2)

    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(x ** 2)
    sumx2 = cum_sumx2[m:n] - cum_sumx2[:n - m]
    sumx = cum_sumx[m:n] - cum_sumx[:n - m]
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - (meanx ** 2)
    sigmax = np.sqrt(sigmax2)

    dist = (sumx2 - 2 * sumx * meanx + m * (meanx ** 2)) / sigmax2 - 2 * (z[m:n] - sumy * meanx) / sigmax + sumy2
    dist = np.sqrt(np.abs(dist))
    dist1 = euclidean_distance(y, x[:len(y)])
    dist = [dist1] + dist
    return dist

