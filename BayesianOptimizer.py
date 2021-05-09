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
        expected improvement at the given point. Returns negative EI so
            minimize works
    """
    y_mean, y_std = gpr.predict(x.reshape(1, -1), return_std=True)
    delta = y_mean - opt
    if y_std <= 1e-10:
        z_score = 1000000
    else:
        z_score = delta/y_std
    return (delta + y_std * norm.pdf(z_score) - delta * norm.cdf(z_score))[0]

def test_optimizer(test_func, bounds, starting_points, kernel, max_iter,
                   optimum=0.0, tol=1e-2, verbose=False):
    """
    Parameters:
        test_func: Function for the optimizer to optimize
        bounds: test_func bounds to optimize on
        starting_points: tuple of lists. First is input values, second is func_values
        kernel: kernel to pass into the BayesianOptimizer
        max_iter: maximum number of model updates for the BayesianOptimizer
        optimum: known optimum of the test function, defaults to 0.0
        tol: tolerance to error in optimum. model is updated until the optimum
            error is less than tol or max_iter is reached
        verbose: Boolean. If True, this function prints the final optimum from
            the BayesianOptimizer
    Returns:
        opt_list: list of calculated optimums for each model update

    """
    inputs = starting_points[0]
    func_vals = starting_points[1]

    Optimizer = BayesianOptimizer(bounds, f=test_func, inputs=inputs.copy(),
                                  vals=func_vals.copy(), explicit_f=False,
                                  kernel=kernel)
    ​
    flag = True
    count = 0
    optimum = 0.0
    opt_list = []
    ​
    while flag and max_iter > count:
        Optimizer.update_model()
        count += 1
        opt_list.append(Optimizer.get_opt()[1])
        if abs(Optimizer.get_opt()[1] - optimum) <= tol:
            flag = False

    if verbose:
        print('opt =', Optimizer.get_opt())

    return opt_list


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
        explicit_f: Boolean. Determines if inputs and vals are used, or if an
            explicit function is evaluated.
    """

    def __init__(self, bounds, f=None, init_num=5, inputs=None, vals=None,
                 kernel=gaussian_process.kernels.Matern(nu=1.5), acq_func=EI,
                 explicit_f=True):
        self._f = f
        self._iterations = 0
        self._bounds = bounds

        if explicit_f: #if an explicit function is given
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

        self._gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2).fit(self._inputs, self._func_vals)
        self._recommended_point = self._inputs[-1]


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


    def update_model(self, new_input=None, acq_func=EI, acq_func_params=None, n_iter=10):
        """
        update model of function with new data and a specified
        Parameters:
            new_input: new data for if there is no explicit function. Should be
                array-like (input, function_value)
            acq_func: acquisition function to optimize in order to find the next
                point to evaluate
            acq_func_params: tuple of the parameters for the acquisition function
                other than a point, the gpr, and the current optimum
            n_iter: maximum number of times to attempt to optimize acq_func
        """
        if new_input is None: #Evaluate function at point given by acquisition func
            self._inputs.append(self._recommended_point)
            self._func_vals.append(self._f(self._recommended_point))
            self._recommended_point = self._acquire_point(acq_func, acq_func_params, n_iter)
        else:
            self._inputs.append(new_input[0])
            self._func_vals.append(new_input[1])
            self._recommended_point = self._acquire_point(acq_func, acq_func_params, n_iter)

        #Update optimum and refit gpr
        id = np.argmax(self._func_vals)
        self._current_opt = (self._inputs[id], self._func_vals[id])
        self._gpr.fit(self._inputs, self._func_vals)

        return True


    def _acquire_point(self, acq_func, acq_func_params, n_iter):
        """
        Choose next point to evaluate based on acquistion function.
        Maximizes the acquisition function to do so.
        Parameters:
            acq_func: acquisition function to use. Defaults to EI
                (expected improvement).
            acq_func_params: tuple of the parameters for the acquisition function
                other than a point, the gpr, and the current optimum
            n_iter: maximum number of times to try optimizing acq_func
                using minimize
        """
        #Expected improvement acquisition function
        max_acq = None
        # Multiply acq_func by -1 so minimize works
        if acq_func_params is None:
            min_acq_func = lambda x, gpr, opt: -acq_func(x, gpr, opt)
        else:
            min_acq_func = lambda x, gpr, opt, params: -acq_func(x, gpr, opt, params)
        for i in range(n_iter):
            seed = uniform(self._bounds[0, :], self._bounds[1, :]).reshape(1, -1)

            # Check if acq_func_params is None, otherwise pass it to the acq_func as
            # param in the minimization
            if acq_func_params is None:
                result = minimize(min_acq_func, seed, method='L-BFGS-B',
                                    bounds=optimize.Bounds(np.array(self._bounds[0, :]),
                                                           np.array(self._bounds[1, :])),
                                    args=(self._gpr, self._current_opt[1]))
            else:
                result = minimize(min_acq_func, seed, method='L-BFGS-B',
                                    bounds=optimize.Bounds(np.array(self._bounds[0, :]),
                                                           np.array(self._bounds[1, :])),
                                    args=(self._gpr, self._current_opt[1], *acq_func_params))
            # Check if optimization was successful, break if it was
            if result.success:
                if max_acq is None or -result.fun >= max_acq:
                    next_eval_point = result.x
                    max_acq = -result.fun
                    break
        # Check if any of the optimizations were successful, if not choose a random point
        if max_acq is None:
            next_eval_point=uniform(self._bounds[0, :], self._bounds[1, :])

        return next_eval_point
