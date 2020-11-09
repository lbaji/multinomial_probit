import numpy as np
from scipy.stats import norm

from numba import njit, guvectorize
from numba.extending import get_cython_function_address
import ctypes

def mc_integration(u_prime, cov, y, n_draws=None):
    """Calculate probit choice probabilities with Monte-Carlo Integration.
    
    Args:
        u_prime (np.array): 2d array of shape (n_obs, n_choices) with the
            deterministic part of utilities
        cov (np.array): 2d array of shape (n_choices - 1, n_choices - 1) with 
            the cov matrix
        y (np.array): 1d array of shape (n_obs) with the observed choices
        n_draws (int): Number of draws for Monte-Carlo integration.
        
    Returns:
        choice_prob_obs (np.array): 1d array of shape (n_obs) with the choice 
        probabilities for the chosen alternative for each individual.
        
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]
    
    if n_draws is None:
        n_draws = n_choices * 2000

    np.random.seed(1995)
    base_error = np.random.normal(size=(n_obs*n_draws, (n_choices - 1)))
    chol = np.linalg.cholesky(cov)
    errors = chol.dot(base_error.T)
    errors = errors.T.reshape(n_obs, n_draws, (n_choices - 1))
    extra_column_errors = np.zeros((n_obs, n_draws, 1))
    errors = np.append(errors, extra_column_errors, axis=2)

    u = u_prime.reshape(n_obs, 1, n_choices) + errors
    
    index_choices = np.argmax(u, axis=2)

    choices = np.zeros((n_obs, n_draws, n_choices))
    for i in range(n_obs):
        for j in range(len(index_choices[1])):
            choices[i, j, int(index_choices[i, j])] = 1

    choice_probs = np.average(choices, axis=1)

    choice_prob_obs = choice_probs[range(len(y)), y]
    
    return choice_prob_obs

def smc_integration(u_prime, cov, y, tau=None, n_draws=None):
    """Calculate probit choice probabilities with smooth Monte-Carlo Integration.
    
    Args:
        u_prime (np.array): 2d array of shape (n_obs, n_choices) with
            deterministic part of utilities
        cov (np.array): 2d array of shape (n_choices - 1, n_choices - 1) with 
            the cov matrix 
        y (np.array): 1d array of shape (n_obs) with the observed choices
        tau (int): corresponds to the smoothing factor. It should be a 
            positive number. For values close to zero the estimated smooth choice 
            probabilities lie in a wider interval which becomes symmetrically 
            smaller for larger values of tau 
        n_draws (int): number of draws for smooth Monte-Carlo integration.
        
                
    Returns:
        choice_prob_obs (np.array): 1d array of shape (n_obs) with the choice 
        probabilities for the chosen alternative for each individual.
        
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]
    huge = 1e250
    
    if n_draws is None:
        n_draws = n_choices * 2000
        
    if tau is None:
        tau = 1
    
    np.random.seed(1965)
    base_error = np.random.normal(size=(n_obs*n_draws, (n_choices - 1)))
    chol = np.linalg.cholesky(cov)
    errors = chol.dot(base_error.T)
    errors = errors.T.reshape(n_obs, n_draws, (n_choices - 1))
    extra_column_errors = np.zeros((n_obs, n_draws, 1))
    errors = np.append(errors, extra_column_errors, axis=2)

    u = u_prime.reshape(n_obs, 1, n_choices) + errors
    
    u_max = np.max(u, axis=2)    
    val_exp = np.clip(np.exp((u - u_max.reshape(n_obs, n_draws, 1)) / tau), 0, huge)    
    smooth_dummy = val_exp / val_exp.sum(axis=2).reshape(n_obs, n_draws, 1)    
    choice_probs = np.average(smooth_dummy, axis=1)
    
    choice_prob_obs = choice_probs[range(len(y)), y]
 
    return choice_prob_obs

def gauss_integration(u_prime, cov, y, degrees=None):
    """Calculate probit choice probabilities with Gauss-Laguerre Integration.
    
    Args:
        u_prime (np.array): 2d array of shape (n_obs, n_choices) with
            deterministic part of utilities
        cov (np.array): 2d array of shape (n_choices - 1, n_choices - 1) with 
            the cov matrix 
        y (np.array): 1d array of shape (n_obs) with the observed choices
        degrees (int): order of polynomial for the approximation.
                
    Returns:
        choice_probs (np.array): 1d array of shape (n_obs) with the choice 
        probabilities for the chosen alternative for each individual.
        
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]
    
    if degrees is None:
        degrees = 25
    x_k, w_k =np.polynomial.laguerre.laggauss(degrees)
    fraction = np.divide(w_k, np.sqrt(x_k))
    sqrt_x_k = np.sqrt(2*x_k)
    
    u_choice = np.choose(y, u_prime.T)
    dif_array = u_prime - u_choice.reshape(n_obs, 1)
    
    u_dif = np.zeros((n_obs, n_choices - 1))
    for counter, choice in enumerate(y):
        u_dif[counter] = np.delete(dif_array[counter, :], choice)
    
    phi_neg_choice = norm.cdf(- sqrt_x_k.reshape(1, degrees, 1) 
                                - u_dif.reshape(n_obs, 1, n_choices - 1)) 
    phi_pos_choice = norm.cdf(sqrt_x_k.reshape(1, degrees, 1) 
                                - u_dif.reshape(n_obs, 1, n_choices -1))
    
    choice_probs_interm = phi_neg_choice.prod(axis=2) + phi_pos_choice.prod(axis=2)
    choice_prob_obs = (1/(2*np.sqrt(np.pi)))*np.dot(choice_probs_interm, fraction)
    
    return choice_prob_obs



# FROM HERE ON THIS CODE IS MADE BY JANOS!!
    
cdf_addr = get_cython_function_address("scipy.special.cython_special", '__pyx_fuse_1ndtr')
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
normcdf = functype(cdf_addr)

ppf_addr = get_cython_function_address("scipy.special.cython_special", 'ndtri')
normppf = functype(ppf_addr)

log_cdf_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1log_ndtr")
log_normcdf = functype(log_cdf_addr)

def get_seed_points(alphas, seeds):
    """Seed points for the generation of r-sequences.
    
    The r-sequences are seeded with a real number between 0 and 1.
    This seed is used to calculate a first point. All other points
    can be calculated recursively from the first point. We call
    this first point a seed_point.
    
    Args:
        alphas (np.array): see https://tinyurl.com/y4mxf5xb
        seeds (np.array): 1d array if floats between 0 and 1
    
    """
    
    
    dim = len(alphas)
    num_seeds = len(seeds)
    res = np.column_stack([seeds] * dim)
    res += alphas
    res = np.mod(res, 1)
    return res
  
def get_alphas(dim):
    """Get alphas for the r-sequences. 
    
    For details see: https://tinyurl.com/y4mxf5xb
    
    """
    phi_d = get_phi(dim)
    exponents = np.array(range(1, dim + 1))
    alphas = 1 / phi_d ** exponents
    return alphas

def get_phi(dim, max_iter=100):
    """Get generalization of golden ratio to dimension dim.
    
    For details see: https://tinyurl.com/y4mxf5xb
    
    """
    val = 1
    for i in range(max_iter):
        val = (1 + val) ** (1 / (dim + 1))
    return val


def get_transformation_matrix(dim, k):
    """Matrix to transform the full cov into choice specific covs.
    
    This assumes that the last choice is used to normalize the location.
    
    """
    m = np.eye(dim)
    m[:, k] = - 1
    m = np.delete(m, k, axis=0)
    m = np.delete(m, dim - 1, axis=1)
    return m


@njit
def drop(drop_pos, arr):
    """Drop element at drop_pos along axis 1.
    
    Args: 
        drop_pos (np.array): 1d array of length nobs
        arr (np.array): 2d array of shape (nobs, m)
        
    Returns:
        res (np.array): 2d array of shape (nobs, m - 1)
    """
    length, width = arr.shape
    res = np.zeros((length, width - 1))
    for i in range(length):
        j_prime = 0
        for j in range(width):
            if j != drop_pos[i]:
                res[i, j_prime] = arr[i, j]
                j_prime += 1
    return res

@guvectorize(
    ["f8[:], f8[:, :, :], f8[:], f8[:], i8, i8, f8[:]"],
    "(dim), (dim_plus_one, dim, dim), (m), (m), (), () -> ()",
    nopython=True,
    target='cpu',
)
def guvec_ghk_cdf(x, chol, seed_point, alphas, num_points, choice, result):
    dim = len(x)
    
    res = 0.0 
    z = np.zeros(dim - 1)
    
    r_point = seed_point.copy()
    
    for i in range(num_points):
        a = x[0]
        j = 0
        a = normcdf(a / chol[choice, j, j])
        p = a
        z[j] = normppf(r_point[j] * a)

        for j in range(1, dim - 1):
            a = x[j]
            for m in range(j):
                a -= z[m] * chol[choice, j, m]
            a = normcdf(a / chol[choice, j, j])
            p *= a
            uni = r_point[j]
            z[j] = normppf(r_point[j] * a)

        j = dim - 1
        a = x[j]
        for m in range(j):
            a -= z[m] * chol[choice, j, m]
        a = normcdf(a / chol[choice, j, j])
        p *= a
        res += p
        
        # update the r_point for the next iteration
        for j in range(dim - 1):
            r_point[j] = np.mod(r_point[j] + alphas[j], 1)
        
    result[0] = res / num_points
    
def mvn_cdf(x, chols, choice, seed=1995, num_points=None):
    """Multivariate-normal cumulative distribution function.
    
    Args:
        x (np.array): 2d array of shape (nobs, m).
            Each row is a point at which the multivariate-normal cdf is evaluated.
        chols (np.array): 3d array of shape (nchoices, m, m) with 
            the cholesky factors of the covariance matrix for each choice. 
        choice (np.array): 1d array with the observed choice.
            It is assumed that choice labels are integers, start at 0 and increase
            in increments of 1.
        seed (int): A seed for the numpy random number generator. This seed is 
            only used to generate seeds for the r-sequences.
        num_points (int): Number of r-sequence points used to approximate the
            choice probabilities of each individual. Default None. If None,
            the number of points is 1000 * m.
            
    Returns:
        probs (np.array): 1d array of length nobs with choice probabilities. 
        
    """
    nobs, m = x.shape
    num_points = int(100 * m) if num_points is None else int(num_points)
    np.random.seed(seed)
    seeds = np.random.uniform(size=nobs)
    alphas = get_alphas(m - 1)
    seed_points = get_seed_points(alphas, seeds)
    return guvec_ghk_cdf(x, chols, seed_points, alphas, num_points, choice)

def mprobit_choice_probabilities(u_prime, cov, choice, num_points=None, seed=1995):
    """Calculate multinomial probit choice probabilities.
    
    The problem is first expressed as an evaluation of a multivariate normal
    distribution function. Then this cdf is evaluated using the GHK algorithm.
    
    Args:
        choice (np.array): 1d array with the observed choice. 
            It is assumed that choice labels are integers, start at 0 and increase
            in increments of 1.
        u_prime (np.array): 2d array of shape (nobs, nchoices) with the deterministic
            part of the utilities. It is assumed that the last column contains zeros. 
        cov (np.array): The (nchoices - 1, nchoices - 1) covariance matrix.
    """
    # extract number of choices
    num_choices = len(cov) + 1
    m = len(cov)
    
    if num_points is None:
        num_points = num_choices * 1000

    # transform u_prime to point at which cdf is evaluated
    u_chosen = np.choose(choice, u_prime.T)
    x = u_chosen.reshape(-1, 1) - drop(choice, u_prime)
    
    # construct transformed covariance matrices (one for each choice)
    covs = np.zeros((num_choices, m, m))
    for c in range(num_choices):
        trans_mat = get_transformation_matrix(num_choices, c)
        covs[c] = trans_mat.dot(cov).dot(trans_mat.T)
        
    # take their cholesky factor
    chols = np.linalg.cholesky(covs)
    
    # call the multivariate normal cdf
    probs = mvn_cdf(x, chols, choice, seed, num_points)
    return probs




