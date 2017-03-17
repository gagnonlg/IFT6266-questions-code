import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def bayes_error(mu1, mu2, sigma):
    xstar = 0.5 * (mu1 + mu2)
    d = np.sqrt(2) * sigma
    cdf_1 = 0.5 * (1 + erf((xstar - mu1)/d))
    cdf_2 = 0.5 * (1 + erf((xstar - mu2)/d)) 
    return 0.5 * (1 - cdf_1 + cdf_2)

mu1 = 0
grid = np.zeros((100,100))
sigma_s = np.linspace(0.01, 1, 100)
mu2_s = np.linspace(mu1 + 0.001, mu1 + 1, 100)
for i, sigma in enumerate(sigma_s):
    for j, mu2 in enumerate(mu2_s):
        grid[i][j] = bayes_error(mu1, mu2, sigma)

ext = [0.001, 1, 0.01, 1]
plt.imshow(grid, origin='lower', interpolation=None, extent=ext, aspect='auto')
plt.xlabel('sigma')
plt.ylabel('mu_2 - mu_1')
plt.colorbar().set_label('Bayes error')
#plt.show()
plt.savefig('bayes.png')
