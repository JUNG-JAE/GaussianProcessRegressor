import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def plot_data(X, Y):
    plt.plot(X, Y, label=r"$f(x) = x \sin(x)$", color="navy", linestyle="dotted")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):

    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.4, color="#ff7f0e", label="Variance")
    plt.plot(X, mu, label='Mean', color="darkorange")
    
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.tight_layout()
    plt.legend(loc='upper left')


def kernel(X1, X2, type="isotropic_squared_exponential", l=1.0, sigma_f=1.0):
    
    if type == "isotropic_squared_exponential":
        """  
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
        Returns:
            (m x n) matrix.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    
    elif type == "squared_exponential":
        theta_1, theta_2 = 1, 10
        return (theta_1 * np.exp(-0.5 * theta_2 * np.subtract.outer(X1, X2)**2)).squeeze()


def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1). 
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s)
    K_ss = kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
    return mu_s, cov_s


# Generate dataset
X = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)
Y = np.squeeze(X * np.sin(X))

# Randomly selected observed data
training_indices = np.random.RandomState(1).choice(np.arange(Y.size), size=6, replace=False)
X_train, Y_train = X[training_indices], Y[training_indices]

""" """
# Mean and covariance of the prior
mu, cov = np.zeros(X.shape), kernel(X, X)
samples = np.random.multivariate_normal(mu.ravel(), cov, 1)
plt.title("Prior")
plot_data(X, Y)
plot_gp(mu, cov, X, samples=samples)

plt.savefig("prior.png")
plt.show()
# Noise free training data
# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(X, X_train, Y_train)
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 1)
plt.title("Posterior")
plot_data(X, Y)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)

plt.savefig("posterior.png")
plt.show()