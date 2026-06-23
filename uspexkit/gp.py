"""Gaussian Process Regression via GPyTorch — sklearn-compatible wrapper.

Replaces sklearn.gaussian_process.GaussianProcessRegressor with a GPyTorch
backend while preserving the same API: fit(), predict(return_std=True),
kernel_, log_marginal_likelihood(), and pickle serialization.

The composite kernel used throughout uspexkit is:

    0.00581² · DotProduct(σ₀=0.412)
  + 0.35²    · Matern(ν=2.5, ARD lengthscales)
  + WhiteKernel(noise)

In GPyTorch this maps to:

    ScaleKernel(LinearKernel()) + ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D))

with observation noise handled by a GaussianLikelihood.
"""

import numpy as np
import torch
import gpytorch


# ── kernel builder ──────────────────────────────────────────────────────────

def _build_kernel(n_features: int) -> gpytorch.kernels.Kernel:
    """Build the gpytorch equivalent of the uspexkit composite kernel.

    sklearn original:
        0.00581² * DotProduct(sigma_0=0.412)
      + 0.35²    * Matern(length_scale=[0.1]*D, nu=2.5)
      + WhiteKernel(noise_level=0.031)

    GPyTorch equivalent (WhiteKernel → likelihood):
        ScaleKernel(LinearKernel()) + ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D))
    """
    linear = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    linear.outputscale = 0.00581 ** 2

    matern = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)
    )
    matern.outputscale = 0.35 ** 2
    matern.base_kernel.lengthscale = 0.1  # broadcast to all dims

    return linear + matern


# ── internal GP model ───────────────────────────────────────────────────────

class _ExactGP(gpytorch.models.ExactGP):
    """Thin ExactGP wrapper used internally by GPyTorchRegressor."""

    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ── sklearn-compatible regressor ────────────────────────────────────────────

class GPyTorchRegressor:
    """Gaussian Process Regressor backed by GPyTorch.

    Drop-in replacement for ``sklearn.gaussian_process.GaussianProcessRegressor``
    with the same public API surface used by uspexkit.

    Parameters
    ----------
    kernel : sklearn kernel, optional
        Accepted for API compatibility; the equivalent gpytorch kernel is
        always built internally.  The WhiteKernel ``noise_level`` is used
        as the initial likelihood noise when available.
    n_restarts_optimizer : int, default=0
        Number of L-BFGS restarts with different initialisations.
    alpha : float, default=1e-10
        Jitter added to the diagonal for numerical stability.
    normalize_y : bool, default=False
        Centre and scale *y* before fitting; predictions are un-scaled.
    training_iter : int, default=200
        Max L-BFGS iterations per restart.
    """

    def __init__(self, kernel=None, n_restarts_optimizer=0, alpha=1e-10,
                 normalize_y=False, random_state=None, training_iter=200):
        self._sklearn_kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.training_iter = training_iter

        # Populated by fit()
        self.kernel_ = None       # gpytorch kernel (for API compat)
        self._model = None
        self._likelihood = None
        self._X_train = None      # torch tensor
        self._y_train = None      # torch tensor
        self._y_mean = 0.0
        self._y_std = 1.0
        self._lml = None
        self._n_features = None

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the GP model.

        Parameters
        ----------
        X : (N, D) array-like
        y : (N,) array-like

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self._n_features = n_features

        # --- normalise y ---
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std < 1e-12:
                self._y_std = 1.0
            y_proc = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_proc = y.copy()

        train_x = torch.tensor(X, dtype=torch.float64)
        train_y = torch.tensor(y_proc, dtype=torch.float64)

        # --- initial noise guess (from sklearn WhiteKernel if available) ---
        init_noise = 0.031
        if self._sklearn_kernel is not None:
            try:
                # sklearn Sum kernel: iterate k1, k2 to find WhiteKernel
                for k in [self._sklearn_kernel.k1, self._sklearn_kernel.k2]:
                    if hasattr(k, 'noise_level'):
                        init_noise = float(k.noise_level)
                        break
            except Exception:
                pass

        # --- multiple restarts ---
        n_restarts = max(1, self.n_restarts_optimizer + 1)
        best_lml = -float('inf')
        best_state = None
        model = None
        likelihood = None

        for _ in range(n_restarts):
            kernel = _build_kernel(n_features)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            )
            likelihood.noise = init_noise

            model = _ExactGP(train_x, train_y, likelihood, kernel)
            model.train()
            likelihood.train()

            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=0.1,
                max_iter=self.training_iter,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            def _closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                return loss

            try:
                with gpytorch.settings.cholesky_jitter(self.alpha):
                    optimizer.step(_closure)
            except Exception:
                continue

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(self.alpha):
                output = model(train_x)
                lml_val = mll(output, train_y).item()

            if lml_val > best_lml:
                best_lml = lml_val
                best_state = {
                    'model': {k: v.clone() for k, v in model.state_dict().items()},
                    'likelihood': {k: v.clone()
                                   for k, v in likelihood.state_dict().items()},
                }

        # --- restore best ---
        if best_state is None:
            best_lml = lml_val
            best_state = {
                'model': {k: v.clone() for k, v in model.state_dict().items()},
                'likelihood': {k: v.clone()
                               for k, v in likelihood.state_dict().items()},
            }

        kernel = _build_kernel(n_features)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = _ExactGP(train_x, train_y, likelihood, kernel)
        self._model.load_state_dict(best_state['model'])
        self._likelihood = likelihood
        self._likelihood.load_state_dict(best_state['likelihood'])
        self._model.eval()
        self._likelihood.eval()

        self._lml = best_lml
        self.kernel_ = self._model.covar_module
        self._X_train = train_x
        self._y_train = train_y

        return self

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self, X, return_std=False):
        """Predict mean (and optionally std) for test points.

        Parameters
        ----------
        X : (M, D) array-like
        return_std : bool

        Returns
        -------
        y_mean : (M,) ndarray
        y_std  : (M,) ndarray  (only when ``return_std=True``)
        """
        X = np.asarray(X, dtype=np.float64)
        test_x = torch.tensor(X, dtype=torch.float64)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(self.alpha):
            posterior = self._model(test_x)
            mean = posterior.mean.numpy()
            std = posterior.stddev.numpy()

        # un-normalise
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std

        if return_std:
            return mean, std
        return mean

    # ── score / LML ──────────────────────────────────────────────────────

    def log_marginal_likelihood(self, theta=None):
        """Log-marginal likelihood of the fitted model."""
        if self._lml is None:
            raise RuntimeError('Model has not been fitted yet.')
        return self._lml

    def score(self, X, y):
        """R² score on *X*, *y*."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    # ── pickle support ───────────────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        # serialise model parameters as CPU state dicts
        if self._model is not None:
            state['_model_state'] = {
                k: v.cpu().clone() for k, v in self._model.state_dict().items()
            }
            state['_likelihood_state'] = {
                k: v.cpu().clone() for k, v in self._likelihood.state_dict().items()
            }
        else:
            state['_model_state'] = None
            state['_likelihood_state'] = None

        # store training data as numpy (torch tensors don't pickle cleanly
        # across processes)
        if self._X_train is not None:
            state['_X_train_np'] = self._X_train.numpy()
            state['_y_train_np'] = self._y_train.numpy()
        else:
            state['_X_train_np'] = None
            state['_y_train_np'] = None

        # drop live torch objects
        for key in ('_model', '_likelihood', 'kernel_', '_X_train', '_y_train'):
            state[key] = None
        return state

    def __setstate__(self, state):
        model_state = state.pop('_model_state', None)
        likelihood_state = state.pop('_likelihood_state', None)
        X_np = state.pop('_X_train_np', None)
        y_np = state.pop('_y_train_np', None)

        self.__dict__.update(state)

        if model_state is not None and self._n_features is not None:
            kernel = _build_kernel(self._n_features)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            if X_np is not None:
                train_x = torch.tensor(X_np, dtype=torch.float64)
                train_y = torch.tensor(y_np, dtype=torch.float64)
            else:
                train_x = torch.zeros(1, self._n_features, dtype=torch.float64)
                train_y = torch.zeros(1, dtype=torch.float64)

            self._model = _ExactGP(train_x, train_y, likelihood, kernel)
            self._model.load_state_dict(model_state)
            self._likelihood = likelihood
            self._likelihood.load_state_dict(likelihood_state)
            self._model.eval()
            self._likelihood.eval()
            self.kernel_ = self._model.covar_module
            self._X_train = train_x
            self._y_train = train_y
