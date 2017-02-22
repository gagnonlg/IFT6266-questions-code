import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal


NTRIALS = 100

def naive_convolution(x,w):
    return np.convolve(x, w)

def fancy_convolution(x, w):
    return scipy.signal.fftconvolve(x, w)

x = np.random.uniform(size=100)
w = np.random.uniform(size=100)
np.testing.assert_allclose(naive_convolution(x,w), fancy_convolution(x, w))


def measure(size, conv):
    x = np.random.uniform(size=size)
    w = np.random.uniform(size=size)
    n_times = []
    for _ in range(NTRIALS):
        t0 = time.time()
        conv(x, w)
        n_times.append(time.time() - t0)
    return np.mean(n_times), np.std(n_times)

def measure_conv(conv):
    sizes = np.arange(100, 1100, 100)
    times = np.zeros_like(sizes, dtype=float)
    times_std = np.zeros_like(sizes, dtype=float)
    for i in range(sizes.shape[0]):
        times[i], times_std[i] = measure(sizes[i], conv)
    return sizes, times, times_std

n_sz, n_t, n_t_s = measure_conv(naive_convolution)
f_sz, f_t, f_t_s = measure_conv(fancy_convolution)

na, _ = scipy.optimize.curve_fit(lambda n, a, b: a * n*n + b, n_sz, n_t, sigma=n_t_s)
fa, _ = scipy.optimize.curve_fit(lambda n, a, b: a * n*np.log(n) + b, f_sz, f_t, sigma=f_t_s)


plt.errorbar(n_sz, n_t, yerr=n_t_s, fmt='ko', markersize=5, label='naive')
plt.errorbar(f_sz, f_t, yerr=f_t_s, fmt='ro', markersize=5, label='fancy')

x = np.linspace(100, 1000, 1000)
plt.plot(x, na[0] * x * x + na[1], 'k--', label='N^2 fit')
plt.plot(x, fa[0] * x * np.log(x) + fa[1], 'r--', label='Nlog(N) fit')

plt.legend(loc='best')
plt.xlabel('N')
plt.ylabel('Time [s]')
#plt.show()
plt.savefig('convolution.png')
