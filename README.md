# Bayesian_Optimizer #

Bayesian Optimizer object, designed to optimize expensive to optimizer expensive
to evaluate functions by fitting a normal distribution of possible fits to the
data using Gaussian process regression, then determining the next best point to
evaluate by maximizing an acquisition function, and refitting the Gaussian
process regressor.

## Current acquisition function options ##
- Expected improvement

## Intended additions ##
- Knowledge gradient
- Entropy search
