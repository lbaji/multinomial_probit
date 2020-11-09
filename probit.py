import numpy as np
import pandas as pd
from scipy.stats import norm
from processing import multinomial_processing
from estimagic.optimization.optimize import maximize
import integration_methods  

#MULTINOMIAL CASE
    
def multinomial_probit_loglikeobs(params, y, x, cov_structure, integration_method):
    """Construct Log-likelihood contribution per individual of a probit model.
    
    Args:
        params (pd.Series): The parameters of the model. The index consists of 
            the parmater names and the values are the parameters
        y (np.array): 1d numpy array of shape (n_obs) with the observed choices
        x (np.array): 2d numpy array of shape (n_obs, n_var) with the independent 
            variables
        cov_structure (str): Takes values "iid" or "free"
        integration_method (str): Takes the values 'mc_integration', 
            'smooth_mc_integration', 'gauss_integration' or 
            'mprobit_choice_probabilities'. 
    Returns:
        loglikeobs (np.array): 1d numpy array of shape (n_obs) with likelihood 
            contribution per individual.
            
    """
    n_var = np.shape(x)[1]
    n_choices = len(np.unique(y))
       
    if cov_structure == 'iid':
        cov = np.eye(n_choices - 1) + np.ones((n_choices - 1, n_choices - 1))
        
    else:
        covariance = params['covariance'].to_numpy()
        
        cov = np.zeros((n_choices - 1, n_choices - 1))

        a = 0
        for i in range(n_choices - 1):
            l = i + 1
            cov[i, :(i+1)] = covariance[a:(a + l)]
            a += l

        for i in range(n_choices - 1):
            cov[i, (i+1):] = cov[(i+1):, i]

    bethas = np.zeros((n_var, n_choices))

    for i in range(n_choices - 1):
        bethas[:, i] = params['choice_{}'.format(i)].to_numpy()
        
    u_prime = x.dot(bethas)
        
    choice_prob_obs = getattr(integration_methods, integration_method)(u_prime, 
                                 cov, y) 
    
    choice_prob_obs[choice_prob_obs<=1e-250] = 1e-250
    
    loglikeobs = np.log(choice_prob_obs)

    return loglikeobs

def multinomial_probit_loglike(params, y, x, cov_structure, integration_method):
    """Construct Log-likelihood of the multinomial probit model.
    
    Args:
        params (pd.Series): The parameters of the model. The index consists of 
            the parmater names and the values are the parameters
        y (np.array): 1d numpy array of shape (n_obs) with the observed choices
        x (np.array): 2d numpy array of shape (n_obs, n_var) with the independent 
            variables
        cov_structure (str): Takes values "iid" or "free"
        integration_method (str): Takes the values 'mc_integration', 
            'smooth_mc_integration', 'gauss_integration' or 
            'mprobit_choice_probabilities'. 
    Returns:
        loglike (float): The loglikelihood of the problem.

    """
    return multinomial_probit_loglikeobs(params, y, x, cov_structure, 
                                         integration_method).sum()
    
def multinomial_probit(formula, data, cov_structure, integration_method, 
                       algorithm):
    """Optimize multinomial probit model.
    
    Args:
        formula (str): A patsy formula
        data (pd.DataFrame): The dataset
        cov_structure (str): Takes values "iid" or "free"
        integration_method (str): Takes the values 'mc_integration', 
            'smooth_mc_integration', 'gauss_integration' or 
            'mprobit_choice_probabilities'. 
        algorithn (str): Takes the values 'scipy_L-BFGS-B', 'scipy_SLSQP', 
            'nlopt_bobyqa' or 'nlopt_newuoa_bound'.
        
    Returns:
        result_dict (dic): Information of the optimization, for example the value
            of the maximized log likelihood
        params: Optimal parameters for the model.
    """
    y, x, params = multinomial_processing(formula, data, cov_structure)
    params_df = pd.DataFrame(params, columns=['value'])
    
    if cov_structure == 'iid':
        constraints = []
    
    else:
        constraints = [
            {'loc': 'covariance', 'type': 'covariance'},
            {'loc': ('covariance', 0), 'type': 'fixed', 'value': 1.0}
        ]
        
    result = maximize(multinomial_probit_loglike, params_df, algorithm, 
                      criterion_kwargs={'y': y, 'x': x, 
                                        'cov_structure': cov_structure, 
                                        'integration_method': integration_method}, 
                                        constraints=constraints, dashboard=False)
    
    return result