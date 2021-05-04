"""
James Carpenter
University of Notre Dame, CBE
March 2021
Bayesian optimizer module
"""

from sklearn import gaussian_process
import numpy as np
from scipy.stats import norm
from numpy.random import uniform
from scipy.optimize import minimize
import scipy.optimize as optimize


def EI(x, gpr, opt):
    """
    Expected improvement acquisition function. Returns negative EI so minimize
    works
    Parameters:
        x: point to evaluate in the Gaussian process regressor
        gpr: Gaussian process regressor object
        opt: current optimal point
    Returns:
        EI: expected improvement at the given point. Returns negative EI so
            minimize works
    """
    y_mean, y_std = gpr.predict(x.reshape(1, -1), return_std=True)
    delta = y_mean - opt
    if y_std <= 1e-10:
        z_score = 1000000
    else:
        z_score = delta/y_std
    return (delta + y_std * norm.pdf(z_score) - delta * norm.cdf(z_score))[0]


class BayesianOptimizer:
    """
    Parameters:
        f: function to optimize, can be left as None if there is no explicit
            function, in which case data must be input
        inputs: input data points if there is no explicit function. Should be a
            list
        vals: list of function values corresponding to inputs if there is no
            explicit function
        bounds: array of bounds for each variable in f. [0, :] should be lower bounds
        init_num: int of points to randomly evaluate initally before approximating
            the function and using the acquisition function. Defaults to 5.
        kernel: a kernel to use in the GPR, pass in a kernel object from
            sklearn.gaussian_process.kernels. Defaults to Matern kernel with nu
            set to 1.5.
    """

    def __init__(self, bounds, f=None, init_num=5, inputs=None, vals=None,
                 kernel=gaussian_process.kernels.Matern(nu=1.5), acq_func=EI):
        self._f = f
        self._iterations = 0
        self._bounds = bounds

        if self._f is not None: #if an explicit function is given
            self._inputs = []
            self._func_vals = [] #List of tuples of function inputs and function outputs
            # Random initial inputs to obtain some function values
            for i in range(init_num):
                inputs = (uniform(bounds[0, :], bounds[1, :]))
                self._inputs.append(inputs)
                self._func_vals.append(self._f(inputs))

        else: #if no explicit function is given
            self._inputs = inputs
            self._func_vals = vals

        id = np.argmax(self._func_vals)
        self._current_opt = (self._inputs[id], self._func_vals[id])

        self._gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(self._inputs, self._func_vals)
        self._recommended_point = self._acquire_point(EI, 10)


    def get_opt(self):
        """
        Getter method for self._current_opt
        """
        return self._current_opt


    def get_next_point(self):
        """
        Getter method for self._recommended_point
        """
        return self._recommended_point


    def update_model(self, new_input=None, acq_func=EI, n_iter=10):
        """
        update model of function with new data
        Parameters:
            new_input: new data for if there is no explicit function. Should be
                array-like (input, function_value)
            acq_func: acquisition function to optimize in order to find the next
                point to evaluate
            n_iter: maximum number of times to attempt to optimize acq_func
        """
        if new_input is None: #Evaluate function at point given by acquisition func
            self._inputs.append(self._recommended_point)
            self._func_vals.append(self._f(self._recommended_point))
            self._recommended_point = self._acquire_point(acq_func, n_iter)
        else:
            self._inputs.append(new_input[0])
            self._func_vals.append(new_input[1])
            self._recommended_point = self._acquire_point(acq_func, n_iter)

        #Update optimum and refit gpr
        id = np.argmax(self._func_vals)
        self._current_opt = (self._inputs[id], self._func_vals[id])
        self._gpr.fit(self._inputs, self._func_vals)

        return True


    def _acquire_point(self, acq_func, n_iter):
        """
        Choose next point to evaluate based on acquistion function.
        Parameters:
            acq_func: acquisition function to use. Defaults to EI
                (expected improvement).
            n_iter: number of seeds to use as an initial guess in acq_func
                optimization
        """
        #Expected improvement acquisition function
        max_acq = None
        # Multiply acq_func by -1 so minimize works
        min_acq_func = lambda x, gpr, opt: -acq_func(x, gpr, opt)
        for i in range(n_iter):
            seed = uniform(self._bounds[0, :], self._bounds[1, :]).reshape(1, -1)
            result = minimize(min_acq_func, seed, method='L-BFGS-B',
                                bounds=optimize.Bounds(np.array(self._bounds[0, :]),
                                                       np.array(self._bounds[1, :])),
                                args=(self._gpr, self._current_opt[1]))
            if result.success:
                if max_acq is None or -result.fun >= max_acq:
                    next_eval_point = result.x
                    max_acq = -result.fun

        return next_eval_point
