=========
Multinomial Probit Model
=========

In this repository you will find these four modules: `convergence_plot`, `integration_methods`, `probit` and `processing`. 

The modules `processing` contains functions for the probit binary and the multinomial cases, while the module `probit`contains only functions for the multinomial case.

PROCESSING:

In this module there are two functions: binary_processing and multinomial_processing. For this replication only the multinomial function is relevant. The `multinomial_processing` function produces the starting values for the calculation of the log likelihood of the problem choices. Its outputs are two arrays: one with the choices and the other with the variables, and a series with the parameters. The parameters contained in this series are optimized in a further step. This is the reason for the inclusion of the argument 'cov_structure': if the errors are assumed to be iid, their covariance will not be contained in the parameter series and will not be optimized. On the other hand, if the iid assumption is relaxed, the covariance is assumed to be a free parameter (contained in the parameter series) and will be optimized.

PROBIT:

In this module you will find three functions: `multinomial_probit_loglikeobs`, `multinomial_probit_loglike` and `multinomial_probit`.

- `multinomial_probit_loglikeobs`:

The function 'multinomial_probit_loglikeobs' calculates the log likelihood of the choices for each observation. For that, it uses different integration_methods according to the characteristics of the problem. The methods are located in the module `integration_methods` described below. For the function `multinomial_probit_loglikeobs` the argument 'cov_structure' is 'only' relevant for the creation of the covariance matrix: if 'cov_structure' equals 'iid', the covariance matrix is assumed to be the identity matrix. If instead 'cov_structure' equals 'free' the covariance matrix is created row-wise from the covariance parameters from the parameter series, passed from the function `multinomial_processing` described above.

- `multinomial_probit_loglike`:

The function `multinomial_probit_loglike` returns the sum of the log likelihoods of the choices of all individuals, i.e. the log likelihood of the given problem.

- `multinomial_probit`:

The function 'multinomial_probit' optimizes the parameters given by the function `multinomial_processing` to maximize the log likelihood of the problem given by `multinomial_probit_loglike`. For the optimization it uses estimagic. If the argument 'cov_structure' equals 'iid', no constraints will be passed for the maximization. On the other hand, for 'cov_structure' equal to 'free', two constrainst will be passed: 1) the entries of the parameter series corresponding to covariance must result in a positive semidefinite matrix such that it serves as a covariance matrix, 2) the first variance is normalized to 1. 

INTEGRATION METHODS

The practical obstacle of multinomial probit models is the calculation of the high dimensional integrals, since they do not have a closed-form solution. This problem is solved with simulations or approximations. The higher the number of choices and variables in the model, the more difficult it is to calculate the choice probabilities. This is the reason for the existence of several simulation methods. 

In the module `integration_methods` there are four functions corresponding to the following methods: Monte Carlo, Smooth Monte Carlo, Gauss, and Geweke-Hajivassiliou-Keane (GHK), the later written by Janos (@janosg) together with all the other functions in this module not mentioned here that are called by the main function 'mprobit_choice_probabilities'. Their precision vary according to the number of choices, variables and observations of the model and a comparison between them is presented in the next section.

- `mc_integration`: 

The Monte Carlo Integration is the most intuitive one but presents the following problems:

    - 'Flat regions': the log likelihood generated with this method has flat regions, i.e., for different parameter values the log likelihood does not change. The consequence is that the optimizer stops thinking it has reached a maximum, when that is actually not the case. 
    - Zero probabilities: having a choice probability of zero for some individual cases that the log likelihood equals minus infinity. 
    - Speed: the number of draws needed for an accurate estimation of the choice probabilities increases with the number of choices, in the function number of draws equals 2000 multiplied by the number of choices. This amount of draws is added for each individual and for a big dataset the calculation of this simulation becomes extremly slow. 
    
- `smc_integration`:

The smooth Monte Carlo Integration solves the problem of the flat areas of the normal Monte Carlo but introduces another complication: the estimated choice probabilities are biased and the bias increases with the smoothing factor, tau. In the section below you will find an example in which this is shown. Notice that the bias is already given by 'small' values of tau, such as 1.

- `gauss_integration`: 

The Gaus-Laguerre integration is an approximation of the high dimensional integrals with polynomials. Its main advantage is its speed but it has to important disadvantages: 1) it requires the strong assumption of iid errors for the transformation of the high dimensional integral into a 1-dimensional one and, 2) by its derivation, it works better with large numbers of choices. 

- `ghk_integration`:

The GHK integration's main advantanges are its speed and its unbiased correction for choice probabilities equal to zero. As mentioned before, this function was written by Janos. This is the link to the blogpost that presents the sequence he uses for his ghk_integration: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/.


CONVERGENCE PLOT

The function 'convergence_plot' generates a graph of the values that the choice probabilities of one (example) individual take according to the number of draws (errors) / degrees (polynomial order) / number of points (of the ghk sequence) and number of choices used for their calculation. 
