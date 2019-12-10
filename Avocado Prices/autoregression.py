# -*- coding: utf-8 -*-
#quote from statsmodels/tsa/statespace/sarimax
from warnings import warn

import numpy as np
import pandas as pd

#from statsmodels.compat.pandas import Appender

from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ValueWarning
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat

from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    companion_matrix, diff, is_invertible, constrain_stationary_univariate,
    unconstrain_stationary_univariate,
    prepare_exog, prepare_trend_spec, prepare_trend_data
)


class SARIMAX(MLEModel):
    def __init__(self, endog, exog=None, order=(1, 0, 0),
                 seasonal_order=(0, 0, 0, 0), trend=None,
                 measurement_error=False, time_varying_regression=False,
                 mle_regression=True, simple_differencing=False,
                 enforce_stationarity=True, enforce_invertibility=True,
                 hamilton_representation=False, concentrate_scale=False,
                 trend_offset=1, use_exact_diffuse=False, dates=None,
                 freq=None, missing='none', **kwargs):

        self._spec = SARIMAXSpecification(
            endog, exog=exog, order=order, seasonal_order=seasonal_order,
            trend=trend, enforce_stationarity=None, enforce_invertibility=None,
            concentrate_scale=concentrate_scale, dates=dates, freq=freq,
            missing=missing)
        self._params = SARIMAXParams(self._spec)

        # Save given orders
        order = self._spec.order
        seasonal_order = self._spec.seasonal_order
        self.order = order
        self.seasonal_order = seasonal_order

        # Model parameters
        self.seasonal_periods = seasonal_order[3]
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.use_exact_diffuse = use_exact_diffuse

        # Enforce non-MLE coefficients if time varying coefficients is
        # specified
        if self.time_varying_regression and self.mle_regression:
            raise ValueError('Models with time-varying regression coefficients'
                             ' must integrate the coefficients as part of the'
                             ' state vector, so that `mle_regression` must'
                             ' be set to False.')

        # Lag polynomials
        self._params.ar_params = -1
        self.polynomial_ar = self._params.ar_poly.coef
        self._polynomial_ar = self.polynomial_ar.copy()

        self._params.ma_params = 1
        self.polynomial_ma = self._params.ma_poly.coef
        self._polynomial_ma = self.polynomial_ma.copy()

        self._params.seasonal_ar_params = -1
        self.polynomial_seasonal_ar = self._params.seasonal_ar_poly.coef
        self._polynomial_seasonal_ar = self.polynomial_seasonal_ar.copy()

        self._params.seasonal_ma_params = 1
        self.polynomial_seasonal_ma = self._params.seasonal_ma_poly.coef
        self._polynomial_seasonal_ma = self.polynomial_seasonal_ma.copy()

        # Deterministic trend polynomial
        self.trend = trend
        self.trend_offset = trend_offset
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._polynomial_trend = self.polynomial_trend.copy()

        # Model orders
        # Note: k_ar, k_ma, k_seasonal_ar, k_seasonal_ma do not include the
        # constant term, so they may be zero.
        # Note: for a typical ARMA(p,q) model, p = k_ar_params = k_ar - 1 and
        # q = k_ma_params = k_ma - 1, although this may not be true for models
        # with arbitrary log polynomials.
        self.k_ar = self._spec.max_ar_order
        self.k_ar_params = self._spec.k_ar_params
        self.k_diff = int(order[1])
        self.k_ma = self._spec.max_ma_order
        self.k_ma_params = self._spec.k_ma_params

        self.k_seasonal_ar = (self._spec.max_seasonal_ar_order *
                              self._spec.seasonal_periods)
        self.k_seasonal_ar_params = self._spec.k_seasonal_ar_params
        self.k_seasonal_diff = int(seasonal_order[1])
        self.k_seasonal_ma = (self._spec.max_seasonal_ma_order *
                              self._spec.seasonal_periods)
        self.k_seasonal_ma_params = self._spec.k_seasonal_ma_params

        # Make internal copies of the differencing orders because if we use
        # simple differencing, then we will need to internally use zeros after
        # the simple differencing has been performed
        self._k_diff = self.k_diff
        self._k_seasonal_diff = self.k_seasonal_diff

        # We can only use the Hamilton representation if differencing is not
        # performed as a part of the state space
        if (self.hamilton_representation and not (self.simple_differencing or
           self._k_diff == self._k_seasonal_diff == 0)):
            raise ValueError('The Hamilton representation is only available'
                             ' for models in which there is no differencing'
                             ' integrated into the state vector. Set'
                             ' `simple_differencing` to True or set'
                             ' `hamilton_representation` to False')

        # Model order
        # (this is used internally in a number of locations)
        self._k_order = max(self.k_ar + self.k_seasonal_ar,
                            self.k_ma + self.k_seasonal_ma + 1)
        if self._k_order == 1 and self.k_ar + self.k_seasonal_ar == 0:
            # Handle time-varying regression
            if self.time_varying_regression:
                self._k_order = 0

        # Exogenous data
        (self.k_exog, exog) = prepare_exog(exog)

        # Redefine mle_regression to be true only if it was previously set to
        # true and there are exogenous regressors
        self.mle_regression = (
            self.mle_regression and exog is not None and self.k_exog > 0
        )
        # State regression is regression with coefficients estimated within
        # the state vector
        self.state_regression = (
            not self.mle_regression and exog is not None and self.k_exog > 0
        )
        # If all we have is a regression (so k_ar = k_ma = 0), then put the
        # error term as measurement error
        if self.state_regression and self._k_order == 0:
            self.measurement_error = True

        # Number of states
        k_states = self._k_order
        if not self.simple_differencing:
            k_states += (self.seasonal_periods * self._k_seasonal_diff +
                         self._k_diff)
        if self.state_regression:
            k_states += self.k_exog

        # Number of positive definite elements of the state covariance matrix
        k_posdef = int(self._k_order > 0)
        # Only have an error component to the states if k_posdef > 0
        self.state_error = k_posdef > 0
        if self.state_regression and self.time_varying_regression:
            k_posdef += self.k_exog

        # Diffuse initialization can be more sensistive to the variance value
        # in the case of state regression, so set a higher than usual default
        # variance
        if self.state_regression:
            kwargs.setdefault('initial_variance', 1e10)

        # Handle non-default loglikelihood burn
        self._loglikelihood_burn = kwargs.get('loglikelihood_burn', None)

        # Number of parameters
        self.k_params = (
            self.k_ar_params + self.k_ma_params +
            self.k_seasonal_ar_params + self.k_seasonal_ma_params +
            self.k_trend +
            self.measurement_error +
            int(not self.concentrate_scale)
        )
        if self.mle_regression:
            self.k_params += self.k_exog

        # We need to have an array or pandas at this point
        self.orig_endog = endog
        self.orig_exog = exog
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)

        # Update the differencing dimensions if simple differencing is applied
        self.orig_k_diff = self._k_diff
        self.orig_k_seasonal_diff = self._k_seasonal_diff
        if (self.simple_differencing and
           (self._k_diff > 0 or self._k_seasonal_diff > 0)):
            self._k_diff = 0
            self._k_seasonal_diff = 0

        # Internally used in several locations
        self._k_states_diff = (
            self._k_diff + self.seasonal_periods * self._k_seasonal_diff
        )

        # Set some model variables now so they will be available for the
        # initialize() method, below
        self.nobs = len(endog)
        self.k_states = k_states
        self.k_posdef = k_posdef

        # Initialize the statespace
        super(SARIMAX, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Set the filter to concentrate out the scale if requested
        if self.concentrate_scale:
            self.ssm.filter_concentrated = True

        # Set as time-varying model if we have time-trend or exog
        if self.k_exog > 0 or len(self.polynomial_trend) > 1:
            self.ssm._time_invariant = False

        # Initialize the fixed components of the statespace model
        self.ssm['design'] = self.initial_design
        self.ssm['state_intercept'] = self.initial_state_intercept
        self.ssm['transition'] = self.initial_transition
        self.ssm['selection'] = self.initial_selection
        if self.concentrate_scale:
            self.ssm['state_cov', 0, 0] = 1.

        # update _init_keys attached by super
        self._init_keys += ['order', 'seasonal_order', 'trend',
                            'measurement_error', 'time_varying_regression',
                            'mle_regression', 'simple_differencing',
                            'enforce_stationarity', 'enforce_invertibility',
                            'hamilton_representation', 'concentrate_scale',
                            'trend_offset'] + list(kwargs.keys())
        # TODO: I think the kwargs or not attached, need to recover from ???

        # Initialize the state
        if self.ssm.initialization is None:
            self.initialize_default()

    def prepare_data(self):
        endog, exog = super(SARIMAX, self).prepare_data()

        # Perform simple differencing if requested
        if (self.simple_differencing and
           (self.orig_k_diff > 0 or self.orig_k_seasonal_diff > 0)):
            # Save the original length
            orig_length = endog.shape[0]
            # Perform simple differencing
            endog = diff(endog.copy(), self.orig_k_diff,
                         self.orig_k_seasonal_diff, self.seasonal_periods)
            if exog is not None:
                exog = diff(exog.copy(), self.orig_k_diff,
                            self.orig_k_seasonal_diff, self.seasonal_periods)

            # Reset the ModelData datasets and cache
            self.data.endog, self.data.exog = (
                self.data._convert_endog_exog(endog, exog))

            # Reset indexes, if provided
            new_length = self.data.endog.shape[0]
            if self.data.row_labels is not None:
                self.data._cache['row_labels'] = (
                    self.data.row_labels[orig_length - new_length:])
            if self._index is not None:
                if self._index_int64:
                    self._index = pd.RangeIndex(start=1, stop=new_length + 1)
                elif self._index_generated:
                    self._index = self._index[:-(orig_length - new_length)]
                else:
                    self._index = self._index[orig_length - new_length:]

        # Reset the nobs
        self.nobs = endog.shape[0]

        # Cache the arrays for calculating the intercept from the trend
        # components
        self._trend_data = prepare_trend_data(
            self.polynomial_trend, self.k_trend, self.nobs, self.trend_offset)

        return endog, exog

    def initialize(self):
        
        super(SARIMAX, self).initialize()

        # Cache the indexes of included polynomial orders (for update below)
        # (but we do not want the index of the constant term, so exclude the
        # first index)
        self._polynomial_ar_idx = np.nonzero(self.polynomial_ar)[0][1:]
        self._polynomial_ma_idx = np.nonzero(self.polynomial_ma)[0][1:]
        self._polynomial_seasonal_ar_idx = np.nonzero(
            self.polynomial_seasonal_ar
        )[0][1:]
        self._polynomial_seasonal_ma_idx = np.nonzero(
            self.polynomial_seasonal_ma
        )[0][1:]

        # Save the indices corresponding to the reduced form lag polynomial
        # parameters in the transition and selection matrices so that they
        # do not have to be recalculated for each update()
        start_row = self._k_states_diff
        end_row = start_row + self.k_ar + self.k_seasonal_ar
        col = self._k_states_diff
        if not self.hamilton_representation:
            self.transition_ar_params_idx = (
                np.s_['transition', start_row:end_row, col]
            )
        else:
            self.transition_ar_params_idx = (
                np.s_['transition', col, start_row:end_row]
            )

        start_row += 1
        end_row = start_row + self.k_ma + self.k_seasonal_ma
        col = 0
        if not self.hamilton_representation:
            self.selection_ma_params_idx = (
                np.s_['selection', start_row:end_row, col]
            )
        else:
            self.design_ma_params_idx = (
                np.s_['design', col, start_row:end_row]
            )

        # Cache indices for exog variances in the state covariance matrix
        if self.state_regression and self.time_varying_regression:
            idx = np.diag_indices(self.k_posdef)
            self._exog_variance_idx = ('state_cov', idx[0][-self.k_exog:],
                                       idx[1][-self.k_exog:])

    def initialize_default(self, approximate_diffuse_variance=None):
        """Initialize default"""
        if approximate_diffuse_variance is None:
            approximate_diffuse_variance = self.ssm.initial_variance
        if self.use_exact_diffuse:
            diffuse_type = 'diffuse'
        else:
            diffuse_type = 'approximate_diffuse'

            # Set the loglikelihood burn parameter, if not given in constructor
            if self._loglikelihood_burn is None:
                k_diffuse_states = self.k_states
                if self.enforce_stationarity:
                    k_diffuse_states -= self._k_order
                self.loglikelihood_burn = k_diffuse_states

        init = Initialization(
            self.k_states,
            approximate_diffuse_variance=approximate_diffuse_variance)

        if self.enforce_stationarity:
            # Differencing operators are at the beginning
            init.set((0, self._k_states_diff), diffuse_type)
            # Stationary component in the middle
            init.set((self._k_states_diff,
                      self._k_states_diff + self._k_order),
                     'stationary')
            # Regression components at the end
            init.set((self._k_states_diff + self._k_order,
                      self._k_states_diff + self._k_order + self.k_exog),
                     diffuse_type)
        # If we're not enforcing a stationarity, then we cannot initialize a
        # stationary component
        else:
            init.set(None, diffuse_type)

        self.ssm.initialization = init

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.r_[
            [1] * self._k_diff,
            ([0] * (self.seasonal_periods - 1) + [1]) * self._k_seasonal_diff,
            [1] * self.state_error, [0] * (self._k_order - 1)
        ]

        if len(design) == 0:
            design = np.r_[0]
        if self.state_regression:
            if self._k_order > 0:
                design = np.c_[
                    np.reshape(
                        np.repeat(design, self.nobs),
                        (design.shape[0], self.nobs)
                    ).T,
                    self.exog
                ].T[None, :, :]
            else:
                design = self.exog.T[None, :, :]
        return design

    @property
    def initial_state_intercept(self):
        """Initial state intercept vector"""
        if self.k_trend > 0:
            state_intercept = np.zeros((self.k_states, self.nobs))
        else:
            state_intercept = np.zeros((self.k_states,))
        return state_intercept

    @property
    def initial_transition(self):
        """Initial transition matrix"""
        transition = np.zeros((self.k_states, self.k_states))

        # Exogenous regressors component
        if self.state_regression:
            start = -self.k_exog
            # T_\beta
            transition[start:, start:] = np.eye(self.k_exog)

            # Autoregressive component
            start = -(self.k_exog + self._k_order)
            end = -self.k_exog if self.k_exog > 0 else None
        else:
            # Autoregressive component
            start = -self._k_order
            end = None

        # T_c
        if self._k_order > 0:
            transition[start:end, start:end] = companion_matrix(self._k_order)
            if self.hamilton_representation:
                transition[start:end, start:end] = np.transpose(
                    companion_matrix(self._k_order)
                )

        # Seasonal differencing component
        # T^*
        if self._k_seasonal_diff > 0:
            seasonal_companion = companion_matrix(self.seasonal_periods).T
            seasonal_companion[0, -1] = 1
            for d in range(self._k_seasonal_diff):
                start = self._k_diff + d * self.seasonal_periods
                end = self._k_diff + (d + 1) * self.seasonal_periods

                # T_c^*
                transition[start:end, start:end] = seasonal_companion

                # i
                for i in range(d + 1, self._k_seasonal_diff):
                    transition[start, end + self.seasonal_periods - 1] = 1

                # \iota
                transition[start, self._k_states_diff] = 1

        # Differencing component
        if self._k_diff > 0:
            idx = np.triu_indices(self._k_diff)
            # T^**
            transition[idx] = 1
            # [0 1]
            if self.seasonal_periods > 0:
                start = self._k_diff
                end = self._k_states_diff
                transition[:self._k_diff, start:end] = (
                    ([0] * (self.seasonal_periods - 1) + [1]) *
                    self._k_seasonal_diff)
            # [1 0]
            column = self._k_states_diff
            transition[:self._k_diff, column] = 1

        return transition

    @property
    def initial_selection(self):
        if not (self.state_regression and self.time_varying_regression):
            if self.k_posdef > 0:
                selection = np.r_[
                    [0] * (self._k_states_diff),
                    [1] * (self._k_order > 0), [0] * (self._k_order - 1),
                    [0] * ((1 - self.mle_regression) * self.k_exog)
                ][:, None]

                if len(selection) == 0:
                    selection = np.zeros((self.k_states, self.k_posdef))
            else:
                selection = np.zeros((self.k_states, 0))
        else:
            selection = np.zeros((self.k_states, self.k_posdef))
            # Typical state variance
            if self._k_order > 0:
                selection[0, 0] = 1
            # Time-varying regression coefficient variances
            for i in range(self.k_exog, 0, -1):
                selection[-i, -i] = 1
        return selection

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (SARIMAXResults, SARIMAXResultsWrapper)}

    @staticmethod
    def _conditional_sum_squares(endog, k_ar, polynomial_ar, k_ma,
                                 polynomial_ma, k_trend=0, trend_data=None,
                                 warning_description=None):
        k = 2 * k_ma
        r = max(k + k_ma, k_ar)

        k_params_ar = 0 if k_ar == 0 else len(polynomial_ar.nonzero()[0]) - 1
        k_params_ma = 0 if k_ma == 0 else len(polynomial_ma.nonzero()[0]) - 1

        residuals = None
        if k_ar + k_ma + k_trend > 0:
            try:
                if k_ma > 0:
                    Y = endog[k:]
                    X = lagmat(endog, k, trim='both')
                    params_ar = np.linalg.pinv(X).dot(Y)
                    residuals = Y - np.dot(X, params_ar)
                Y = endog[r:]

                X = np.empty((Y.shape[0], 0))
                if k_trend > 0:
                    if trend_data is None:
                        raise ValueError('Trend data must be provided if'
                                         ' `k_trend` > 0.')
                    X = np.c_[X, trend_data[:(-r if r > 0 else None), :]]
                if k_ar > 0:
                    cols = polynomial_ar.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(endog, k_ar)[r:, cols]]
                if k_ma > 0:
                    cols = polynomial_ma.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(residuals, k_ma)[r-k:, cols]]
                params = np.linalg.pinv(X).dot(Y)
                residuals = Y - np.dot(X, params)
            except ValueError:
                if warning_description is not None:
                    warning_description = ' for %s' % warning_description
                else:
                    warning_description = ''
                warn('Too few observations to estimate starting parameters%s.'
                     ' All parameters except for variances will be set to'
                     ' zeros.' % warning_description)
                params = np.zeros(k_trend + k_ar + k_ma, dtype=endog.dtype)
                if len(endog) == 0:
                    residuals = np.ones(k_params_ma * 2 + 1, dtype=endog.dtype)
                else:
                    residuals = np.r_[
                        np.zeros(k_params_ma * 2, dtype=endog.dtype),
                        endog - np.mean(endog)]

        # Default output
        params_trend = []
        params_ar = []
        params_ma = []
        params_variance = []

        # Get the params
        offset = 0
        if k_trend > 0:
            params_trend = params[offset:k_trend + offset]
            offset += k_trend
        if k_ar > 0:
            params_ar = params[offset:k_params_ar + offset]
            offset += k_params_ar
        if k_ma > 0:
            params_ma = params[offset:k_params_ma + offset]
            offset += k_params_ma
        if residuals is not None:
            params_variance = (residuals[k_params_ma:]**2).mean()

        return (params_trend, params_ar, params_ma,
                params_variance)

    @property
    def start_params(self):
        trend_data = self._trend_data
        if not self.simple_differencing and (
           self._k_diff > 0 or self._k_seasonal_diff > 0):
            endog = diff(self.endog, self._k_diff,
                         self._k_seasonal_diff, self.seasonal_periods)
            if self.exog is not None:
                exog = diff(self.exog, self._k_diff,
                            self._k_seasonal_diff, self.seasonal_periods)
            else:
                exog = None
            trend_data = trend_data[:endog.shape[0], :]
        else:
            endog = self.endog.copy()
            exog = self.exog.copy() if self.exog is not None else None
        endog = endog.squeeze()
        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]
            if trend_data is not None:
                trend_data = trend_data[mask]
        params_exog = []
        if self.k_exog > 0:
            params_exog = np.linalg.pinv(exog).dot(endog)
            endog = endog - np.dot(exog, params_exog)
        if self.state_regression:
            params_exog = []
        (params_trend, params_ar, params_ma,
         params_variance) = self._conditional_sum_squares(
            endog, self.k_ar, self.polynomial_ar, self.k_ma,
            self.polynomial_ma, self.k_trend, trend_data,
            warning_description='ARMA and trend')
        invalid_ar = (
            self.k_ar > 0 and
            self.enforce_stationarity and
            not is_invertible(np.r_[1, -params_ar])
        )
        if invalid_ar:
            warn('Non-stationary starting autoregressive parameters'
                 ' found. Using zeros as starting parameters.')
            params_ar *= 0
        invalid_ma = (
            self.k_ma > 0 and
            self.enforce_invertibility and
            not is_invertible(np.r_[1, params_ma])
        )
        if invalid_ma:
            warn('Non-invertible starting MA parameters found.'
                 ' Using zeros as starting parameters.')
            params_ma *= 0

        # Seasonal Parameters
        _, params_seasonal_ar, params_seasonal_ma, params_seasonal_variance = (
            self._conditional_sum_squares(
                endog, self.k_seasonal_ar, self.polynomial_seasonal_ar,
                self.k_seasonal_ma, self.polynomial_seasonal_ma,
                warning_description='seasonal ARMA'))

        invalid_seasonal_ar = (
            self.k_seasonal_ar > 0 and
            self.enforce_stationarity and
            not is_invertible(np.r_[1, -params_seasonal_ar])
        )
        if invalid_seasonal_ar:
            warn('Non-stationary starting seasonal autoregressive'
                 ' Using zeros as starting parameters.')
            params_seasonal_ar *= 0

        invalid_seasonal_ma = (
            self.k_seasonal_ma > 0 and
            self.enforce_invertibility and
            not is_invertible(np.r_[1, params_seasonal_ma])
        )
        if invalid_seasonal_ma:
            warn('Non-invertible starting seasonal moving average'
                 ' Using zeros as starting parameters.')
            params_seasonal_ma *= 0

        params_exog_variance = []
        if self.state_regression and self.time_varying_regression:
            params_exog_variance = [1] * self.k_exog
        if (self.state_error and type(params_variance) == list and
                len(params_variance) == 0):
            if not (type(params_seasonal_variance) == list and
                    len(params_seasonal_variance) == 0):
                params_variance = params_seasonal_variance
            elif self.k_exog > 0:
                params_variance = np.inner(endog, endog)
            else:
                params_variance = np.inner(endog, endog) / self.nobs
        params_measurement_variance = 1 if self.measurement_error else []

        # We want to bound the starting variance away from zero
        params_variance = np.atleast_1d(max(np.array(params_variance), 1e-10))

        # Remove state variance as parameter if scale is concentrated out
        if self.concentrate_scale:
            params_variance = []

        # Combine all parameters
        return np.r_[
            params_trend,
            params_exog,
            params_ar,
            params_ma,
            params_seasonal_ar,
            params_seasonal_ma,
            params_exog_variance,
            params_measurement_variance,
            params_variance
        ]

    @property
    def endog_names(self, latex=False):
        """Names of endogenous variables"""
        diff = ''
        if self.k_diff > 0:
            if self.k_diff == 1:
                diff = r'\Delta' if latex else 'D'
            else:
                diff = (r'\Delta^%d' if latex else 'D%d') % self.k_diff

        seasonal_diff = ''
        if self.k_seasonal_diff > 0:
            if self.k_seasonal_diff == 1:
                seasonal_diff = ((r'\Delta_%d' if latex else 'DS%d') %
                                 (self.seasonal_periods))
            else:
                seasonal_diff = ((r'\Delta_%d^%d' if latex else 'D%dS%d') %
                                 (self.k_seasonal_diff, self.seasonal_periods))
        endog_diff = self.simple_differencing
        if endog_diff and self.k_diff > 0 and self.k_seasonal_diff > 0:
            return (('%s%s %s' if latex else '%s.%s.%s') %
                    (diff, seasonal_diff, self.data.ynames))
        elif endog_diff and self.k_diff > 0:
            return (('%s %s' if latex else '%s.%s') %
                    (diff, self.data.ynames))
        elif endog_diff and self.k_seasonal_diff > 0:
            return (('%s %s' if latex else '%s.%s') %
                    (seasonal_diff, self.data.ynames))
        else:
            return self.data.ynames

    params_complete = [
        'trend', 'exog', 'ar', 'ma', 'seasonal_ar', 'seasonal_ma',
        'exog_variance', 'measurement_variance', 'variance'
    ]

    @property
    def param_terms(self):
        model_orders = self.model_orders
        # Get basic list from model orders
        params = [
            order for order in self.params_complete
            if model_orders[order] > 0
        ]
        # k_exog may be positive without associated parameters if it is in the
        # state vector
        if 'exog' in params and not self.mle_regression:
            params.remove('exog')

        return params

    @property
    def param_names(self):
        params_sort_order = self.param_terms
        model_names = self.model_names
        return [
            name for param in params_sort_order for name in model_names[param]
        ]

    @property
    def state_names(self):
        k_ar_states = self._k_order
        if not self.simple_differencing:
            k_ar_states += (self.seasonal_periods * self._k_seasonal_diff +
                            self._k_diff)
        names = ['state.%d' % i for i in range(k_ar_states)]

        if self.k_exog > 0 and self.state_regression:
            names += ['beta.%s' % self.exog_names[i]
                      for i in range(self.k_exog)]

        return names

    @property
    def model_orders(self):
        """
        The orders of each of the polynomials in the model.
        """
        return {
            'trend': self.k_trend,
            'exog': self.k_exog,
            'ar': self.k_ar,
            'ma': self.k_ma,
            'seasonal_ar': self.k_seasonal_ar,
            'seasonal_ma': self.k_seasonal_ma,
            'reduced_ar': self.k_ar + self.k_seasonal_ar,
            'reduced_ma': self.k_ma + self.k_seasonal_ma,
            'exog_variance': self.k_exog if (
                self.state_regression and self.time_varying_regression) else 0,
            'measurement_variance': int(self.measurement_error),
            'variance': int(self.state_error and not self.concentrate_scale),
        }

    @property
    def model_names(self):
        """
        The plain text names of all possible model parameters.
        """
        return self._get_model_names(latex=False)

    @property
    def model_latex_names(self):
        """
        The latex names of all possible model parameters.
        """
        return self._get_model_names(latex=True)

    def _get_model_names(self, latex=False):
        names = {
            'trend': None,
            'exog': None,
            'ar': None,
            'ma': None,
            'seasonal_ar': None,
            'seasonal_ma': None,
            'reduced_ar': None,
            'reduced_ma': None,
            'exog_variance': None,
            'measurement_variance': None,
            'variance': None,
        }

        # Trend
        if self.k_trend > 0:
            trend_template = 't_%d' if latex else 'trend.%d'
            names['trend'] = []
            for i in self.polynomial_trend.nonzero()[0]:
                if i == 0:
                    names['trend'].append('intercept')
                elif i == 1:
                    names['trend'].append('drift')
                else:
                    names['trend'].append(trend_template % i)

        # Exogenous coefficients
        if self.k_exog > 0:
            names['exog'] = self.exog_names

        # Autoregressive
        if self.k_ar > 0:
            ar_template = '$\\phi_%d$' if latex else 'ar.L%d'
            names['ar'] = []
            for i in self.polynomial_ar.nonzero()[0][1:]:
                names['ar'].append(ar_template % i)

        # Moving Average
        if self.k_ma > 0:
            ma_template = '$\\theta_%d$' if latex else 'ma.L%d'
            names['ma'] = []
            for i in self.polynomial_ma.nonzero()[0][1:]:
                names['ma'].append(ma_template % i)

        # Seasonal Autoregressive
        if self.k_seasonal_ar > 0:
            seasonal_ar_template = (
                '$\\tilde \\phi_%d$' if latex else 'ar.S.L%d'
            )
            names['seasonal_ar'] = []
            for i in self.polynomial_seasonal_ar.nonzero()[0][1:]:
                names['seasonal_ar'].append(seasonal_ar_template % i)

        # Seasonal Moving Average
        if self.k_seasonal_ma > 0:
            seasonal_ma_template = (
                '$\\tilde \\theta_%d$' if latex else 'ma.S.L%d'
            )
            names['seasonal_ma'] = []
            for i in self.polynomial_seasonal_ma.nonzero()[0][1:]:
                names['seasonal_ma'].append(seasonal_ma_template % i)

        # Reduced Form Autoregressive
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            reduced_polynomial_ar = reduced_polynomial_ar = -np.polymul(
                self.polynomial_ar, self.polynomial_seasonal_ar
            )
            ar_template = '$\\Phi_%d$' if latex else 'ar.R.L%d'
            names['reduced_ar'] = []
            for i in reduced_polynomial_ar.nonzero()[0][1:]:
                names['reduced_ar'].append(ar_template % i)

        # Reduced Form Moving Average
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self.polynomial_ma, self.polynomial_seasonal_ma
            )
            ma_template = '$\\Theta_%d$' if latex else 'ma.R.L%d'
            names['reduced_ma'] = []
            for i in reduced_polynomial_ma.nonzero()[0][1:]:
                names['reduced_ma'].append(ma_template % i)

        # Exogenous variances
        if self.state_regression and self.time_varying_regression:
            if not self.concentrate_scale:
                exog_var_template = ('$\\sigma_\\text{%s}^2$' if latex
                                     else 'var.%s')
            else:
                exog_var_template = (
                    '$\\sigma_\\text{%s}^2 / \\sigma_\\zeta^2$' if latex
                    else 'snr.%s')
            names['exog_variance'] = [
                exog_var_template % exog_name for exog_name in self.exog_names
            ]

        # Measurement error variance
        if self.measurement_error:
            if not self.concentrate_scale:
                meas_var_tpl = (
                    '$\\sigma_\\eta^2$' if latex else 'var.measurement_error')
            else:
                meas_var_tpl = (
                    '$\\sigma_\\eta^2 / \\sigma_\\zeta^2$' if latex
                    else 'snr.measurement_error')
            names['measurement_variance'] = [meas_var_tpl]

        # State variance
        if self.state_error and not self.concentrate_scale:
            var_tpl = '$\\sigma_\\zeta^2$' if latex else 'sigma2'
            names['variance'] = [var_tpl]

        return names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Used primarily to enforce stationarity of the autoregressive lag
        polynomial, invertibility of the moving average lag polynomial, and
        positive variance parameters.

        Parameters
        ----------
        unconstrained : array_like
            Unconstrained parameters used by the optimizer.

        Returns
        -------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self.k_trend > 0:
            end += self.k_trend
            constrained[start:end] = unconstrained[start:end]
            start += self.k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self.k_exog
            constrained[start:end] = unconstrained[start:end]
            start += self.k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            if self.enforce_stationarity:
                constrained[start:end] = (
                    constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            if self.enforce_invertibility:
                constrained[start:end] = (
                    -constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            if self.enforce_stationarity:
                constrained[start:end] = (
                    constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            if self.enforce_invertibility:
                constrained[start:end] = (
                    -constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ma_params

        # Transform the standard deviation parameters to be positive
        if self.state_regression and self.time_varying_regression:
            end += self.k_exog
            constrained[start:end] = unconstrained[start:end]**2
            start += self.k_exog
        if self.measurement_error:
            constrained[start] = unconstrained[start]**2
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            constrained[start] = unconstrained[start]**2
            # start += 1
            # end += 1

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Used primarily to reverse enforcement of stationarity of the
        autoregressive lag polynomial and invertibility of the moving average
        lag polynomial.

        Parameters
        ----------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Returns
        -------
        constrained : array_like
            Unconstrained parameters used by the optimizer.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, constrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self.k_trend > 0:
            end += self.k_trend
            unconstrained[start:end] = constrained[start:end]
            start += self.k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self.k_exog
            unconstrained[start:end] = constrained[start:end]
            start += self.k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            if self.enforce_stationarity:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            if self.enforce_invertibility:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(-constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            if self.enforce_stationarity:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            if self.enforce_invertibility:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(-constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ma_params

        # Untransform the standard deviation
        if self.state_regression and self.time_varying_regression:
            end += self.k_exog
            unconstrained[start:end] = constrained[start:end]**0.5
            start += self.k_exog
        if self.measurement_error:
            unconstrained[start] = constrained[start]**0.5
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            unconstrained[start] = constrained[start]**0.5
            # start += 1
            # end += 1

        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super(SARIMAX, self)._validate_can_fix_params(param_names)
        model_names = self.model_names

        items = [
            ('ar', 'autoregressive', self.enforce_stationarity,
                '`enforce_stationarity=True`'),
            ('seasonal_ar', 'seasonal autoregressive',
                self.enforce_stationarity, '`enforce_stationarity=True`'),
            ('ma', 'moving average', self.enforce_invertibility,
                '`enforce_invertibility=True`'),
            ('seasonal_ma', 'seasonal moving average',
                self.enforce_invertibility, '`enforce_invertibility=True`')]

        for name, title, condition, condition_desc in items:
            names = set(model_names[name] or [])
            fix_all = param_names.issuperset(names)
            fix_any = len(param_names.intersection(names)) > 0
            if condition and fix_any and not fix_all:
                raise ValueError('Cannot fix individual %s parameters when'
                                 ' %s. Must either fix all %s parameters or'
                                 ' none.' % (title, condition_desc, title))

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        params_trend = None
        params_exog = None
        params_ar = None
        params_ma = None
        params_seasonal_ar = None
        params_seasonal_ma = None
        params_exog_variance = None
        params_measurement_variance = None
        params_variance = None

        # Extract the parameters
        start = end = 0
        end += self.k_trend
        params_trend = params[start:end]
        start += self.k_trend
        if self.mle_regression:
            end += self.k_exog
            params_exog = params[start:end]
            start += self.k_exog
        end += self.k_ar_params
        params_ar = params[start:end]
        start += self.k_ar_params
        end += self.k_ma_params
        params_ma = params[start:end]
        start += self.k_ma_params
        end += self.k_seasonal_ar_params
        params_seasonal_ar = params[start:end]
        start += self.k_seasonal_ar_params
        end += self.k_seasonal_ma_params
        params_seasonal_ma = params[start:end]
        start += self.k_seasonal_ma_params
        if self.state_regression and self.time_varying_regression:
            end += self.k_exog
            params_exog_variance = params[start:end]
            start += self.k_exog
        if self.measurement_error:
            params_measurement_variance = params[start]
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            params_variance = params[start]
        # start += 1
        # end += 1

        # Update lag polynomials
        if self.k_ar > 0:
            if self._polynomial_ar.dtype == params.dtype:
                self._polynomial_ar[self._polynomial_ar_idx] = -params_ar
            else:
                polynomial_ar = self._polynomial_ar.real.astype(params.dtype)
                polynomial_ar[self._polynomial_ar_idx] = -params_ar
                self._polynomial_ar = polynomial_ar

        if self.k_ma > 0:
            if self._polynomial_ma.dtype == params.dtype:
                self._polynomial_ma[self._polynomial_ma_idx] = params_ma
            else:
                polynomial_ma = self._polynomial_ma.real.astype(params.dtype)
                polynomial_ma[self._polynomial_ma_idx] = params_ma
                self._polynomial_ma = polynomial_ma

        if self.k_seasonal_ar > 0:
            idx = self._polynomial_seasonal_ar_idx
            if self._polynomial_seasonal_ar.dtype == params.dtype:
                self._polynomial_seasonal_ar[idx] = -params_seasonal_ar
            else:
                polynomial_seasonal_ar = (
                    self._polynomial_seasonal_ar.real.astype(params.dtype)
                )
                polynomial_seasonal_ar[idx] = -params_seasonal_ar
                self._polynomial_seasonal_ar = polynomial_seasonal_ar

        if self.k_seasonal_ma > 0:
            idx = self._polynomial_seasonal_ma_idx
            if self._polynomial_seasonal_ma.dtype == params.dtype:
                self._polynomial_seasonal_ma[idx] = params_seasonal_ma
            else:
                polynomial_seasonal_ma = (
                    self._polynomial_seasonal_ma.real.astype(params.dtype)
                )
                polynomial_seasonal_ma[idx] = params_seasonal_ma
                self._polynomial_seasonal_ma = polynomial_seasonal_ma

        # Get the reduced form lag polynomial terms by multiplying the regular
        # and seasonal lag polynomials
        # Note: that although the numpy np.polymul examples assume that they
        # are ordered from highest degree to lowest, whereas our are from
        # lowest to highest, it does not matter.
        if self.k_seasonal_ar > 0:
            reduced_polynomial_ar = -np.polymul(
                self._polynomial_ar, self._polynomial_seasonal_ar
            )
        else:
            reduced_polynomial_ar = -self._polynomial_ar
        if self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self._polynomial_ma, self._polynomial_seasonal_ma
            )
        else:
            reduced_polynomial_ma = self._polynomial_ma

        if self.mle_regression:
            self.ssm['obs_intercept'] = np.dot(self.exog, params_exog)[None, :]

        if self.k_trend > 0:
            data = np.dot(self._trend_data, params_trend).astype(params.dtype)
            if not self.hamilton_representation:
                self.ssm['state_intercept', self._k_states_diff, :] = data
            else:
                if self.hamilton_representation:
                    data /= np.sum(-reduced_polynomial_ar)

                # If we already set the observation intercept for MLE
                # regression, just add to it
                if self.mle_regression:
                    self.ssm.obs_intercept += data[None, :]
                # Otherwise set it directly
                else:
                    self.ssm['obs_intercept'] = data[None, :]

        # Observation covariance matrix
        if self.measurement_error:
            self.ssm['obs_cov', 0, 0] = params_measurement_variance

        # Transition matrix
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            self.ssm[self.transition_ar_params_idx] = reduced_polynomial_ar[1:]
        elif not self.ssm.transition.dtype == params.dtype:
            self.ssm['transition'] = self.ssm['transition'].real.astype(
                params.dtype)

        # Selection matrix (Harvey) or Design matrix (Hamilton)
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            if not self.hamilton_representation:
                self.ssm[self.selection_ma_params_idx] = (
                    reduced_polynomial_ma[1:]
                )
            else:
                self.ssm[self.design_ma_params_idx] = reduced_polynomial_ma[1:]

        # State covariance matrix
        if self.k_posdef > 0:
            if not self.concentrate_scale:
                self['state_cov', 0, 0] = params_variance
            if self.state_regression and self.time_varying_regression:
                self.ssm[self._exog_variance_idx] = params_exog_variance

        return params


class SARIMAXResults(MLEResults):
    def __init__(self, model, params, filter_results, cov_type=None,
                 **kwargs):
        super(SARIMAXResults, self).__init__(model, params, filter_results,
                                             cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        # Save model specification
        self.specification = Bunch(**{
            # Set additional model parameters
            'seasonal_periods': self.model.seasonal_periods,
            'measurement_error': self.model.measurement_error,
            'time_varying_regression': self.model.time_varying_regression,
            'simple_differencing': self.model.simple_differencing,
            'enforce_stationarity': self.model.enforce_stationarity,
            'enforce_invertibility': self.model.enforce_invertibility,
            'hamilton_representation': self.model.hamilton_representation,
            'concentrate_scale': self.model.concentrate_scale,
            'trend_offset': self.model.trend_offset,

            'order': self.model.order,
            'seasonal_order': self.model.seasonal_order,

            # Model order
            'k_diff': self.model.k_diff,
            'k_seasonal_diff': self.model.k_seasonal_diff,
            'k_ar': self.model.k_ar,
            'k_ma': self.model.k_ma,
            'k_seasonal_ar': self.model.k_seasonal_ar,
            'k_seasonal_ma': self.model.k_seasonal_ma,

            # Param Numbers
            'k_ar_params': self.model.k_ar_params,
            'k_ma_params': self.model.k_ma_params,

            # Trend / Regression
            'trend': self.model.trend,
            'k_trend': self.model.k_trend,
            'k_exog': self.model.k_exog,

            'mle_regression': self.model.mle_regression,
            'state_regression': self.model.state_regression,
        })

        # Polynomials
        self.polynomial_trend = self.model._polynomial_trend
        self.polynomial_ar = self.model._polynomial_ar
        self.polynomial_ma = self.model._polynomial_ma
        self.polynomial_seasonal_ar = self.model._polynomial_seasonal_ar
        self.polynomial_seasonal_ma = self.model._polynomial_seasonal_ma
        self.polynomial_reduced_ar = np.polymul(
            self.polynomial_ar, self.polynomial_seasonal_ar
        )
        self.polynomial_reduced_ma = np.polymul(
            self.polynomial_ma, self.polynomial_seasonal_ma
        )

        # Distinguish parameters
        self.model_orders = self.model.model_orders
        self.param_terms = self.model.param_terms
        start = end = 0
        for name in self.param_terms:
            if name == 'ar':
                k = self.model.k_ar_params
            elif name == 'ma':
                k = self.model.k_ma_params
            elif name == 'seasonal_ar':
                k = self.model.k_seasonal_ar_params
            elif name == 'seasonal_ma':
                k = self.model.k_seasonal_ma_params
            else:
                k = self.model_orders[name]
            end += k
            setattr(self, '_params_%s' % name, self.params[start:end])
            start += k

        # Handle removing data
        self._data_attr_model.extend(['orig_endog', 'orig_exog'])

    def extend(self, endog, exog=None, **kwargs):
        kwargs.setdefault('trend_offset', self.nobs + 1)
        return super(SARIMAXResults, self).extend(endog, exog=exog, **kwargs)

    @cache_readonly
    def arroots(self):
        """
        (array) Roots of the reduced form autoregressive lag polynomial
        """
        return np.roots(self.polynomial_reduced_ar)**-1

    @cache_readonly
    def maroots(self):
        """
        (array) Roots of the reduced form moving average lag polynomial
        """
        return np.roots(self.polynomial_reduced_ma)**-1

    @cache_readonly
    def arfreq(self):
        """
        (array) Frequency of the roots of the reduced form autoregressive
        lag polynomial
        """
        z = self.arroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def mafreq(self):
        """
        (array) Frequency of the roots of the reduced form moving average
        lag polynomial
        """
        z = self.maroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def arparams(self):
        return self._params_ar

    @cache_readonly
    def seasonalarparams(self):
        return self._params_seasonal_ar

    @cache_readonly
    def maparams(self):
        return self._params_ma

    @cache_readonly
    def seasonalmaparams(self):
        return self._params_seasonal_ma

    def get_prediction(self, start=None, end=None, dynamic=False, index=None,
                       exog=None, **kwargs):
        if start is None:
            if (self.model.simple_differencing and
                    not self.model._index_generated and
                    not self.model._index_dates):
                start = 0
            else:
                start = self.model._index[0]

        # Handle start, end, dynamic
        _start, _end, _out_of_sample, prediction_index = (
            self.model._get_prediction_index(start, end, index, silent=True))

        # Handle exogenous parameters
        if _out_of_sample and (self.model.k_exog + self.model.k_trend > 0):
            # Create a new faux SARIMAX model for the extended dataset
            nobs = self.model.data.orig_endog.shape[0] + _out_of_sample
            endog = np.zeros((nobs, self.model.k_endog))

            if self.model.k_exog > 0:
                if exog is None:
                    raise ValueError('Out-of-sample forecasting in a model'
                                     ' with a regression component requires'
                                     ' additional exogenous values via the'
                                     ' `exog` argument.')
                exog = np.array(exog)
                required_exog_shape = (_out_of_sample, self.model.k_exog)
                try:
                    exog = exog.reshape(required_exog_shape)
                except ValueError:
                    raise ValueError('Provided exogenous values are not of the'
                                     ' appropriate shape. Required %s, got %s.'
                                     % (str(required_exog_shape),
                                        str(exog.shape)))
                exog = np.c_[self.model.data.orig_exog.T, exog.T].T

            model_kwargs = self._init_kwds.copy()
            model_kwargs['exog'] = exog
            model = SARIMAX(endog, **model_kwargs)
            model.update(self.params, transformed=True, includes_fixed=True)

            # Set the kwargs with the update time-varying state space
            # representation matrices
            for name in self.filter_results.shapes.keys():
                if name == 'obs':
                    continue
                mat = getattr(model.ssm, name)
                if mat.shape[-1] > 1:
                    kwargs[name] = mat[..., -_out_of_sample:]
        elif self.model.k_exog == 0 and exog is not None:
            warn('Exogenous array provided to predict, but additional data not'
                 ' required. `exog` argument ignored.', ValueWarning)

        return super(SARIMAXResults, self).get_prediction(
            start=start, end=end, dynamic=dynamic, index=index, exog=exog,
            **kwargs)

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None):
        # Create the model name

        # See if we have an ARIMA component
        order = ''
        if self.model.k_ar + self.model.k_diff + self.model.k_ma > 0:
            if self.model.k_ar == self.model.k_ar_params:
                order_ar = self.model.k_ar
            else:
                order_ar = list(self.model._spec.ar_lags)
            if self.model.k_ma == self.model.k_ma_params:
                order_ma = self.model.k_ma
            else:
                order_ma = list(self.model._spec.ma_lags)
            # If there is simple differencing, then that is reflected in the
            # dependent variable name
            k_diff = 0 if self.model.simple_differencing else self.model.k_diff
            order = '(%s, %d, %s)' % (order_ar, k_diff, order_ma)
        # See if we have an SARIMA component
        seasonal_order = ''
        has_seasonal = (
            self.model.k_seasonal_ar +
            self.model.k_seasonal_diff +
            self.model.k_seasonal_ma
        ) > 0
        if has_seasonal:
            tmp = int(self.model.k_seasonal_ar / self.model.seasonal_periods)
            if tmp == self.model.k_seasonal_ar_params:
                order_seasonal_ar = (
                    int(self.model.k_seasonal_ar / self.model.seasonal_periods)
                )
            else:
                order_seasonal_ar = list(self.model._spec.seasonal_ar_lags)
            tmp = int(self.model.k_seasonal_ma / self.model.seasonal_periods)
            if tmp == self.model.k_ma_params:
                order_seasonal_ma = tmp
            else:
                order_seasonal_ma = list(self.model._spec.seasonal_ma_lags)
            # If there is simple differencing, then that is reflected in the
            # dependent variable name
            k_seasonal_diff = self.model.k_seasonal_diff
            if self.model.simple_differencing:
                k_seasonal_diff = 0
            seasonal_order = ('(%s, %d, %s, %d)' %
                              (str(order_seasonal_ar), k_seasonal_diff,
                               str(order_seasonal_ma),
                               self.model.seasonal_periods))
            if not order == '':
                order += 'x'
        model_name = (
            '%s%s%s' % (self.model.__class__.__name__, order, seasonal_order)
            )
        return super(SARIMAXResults, self).summary(
            alpha=alpha, start=start, title='SARIMAX Results',
            model_name=model_name
        )


class SARIMAXResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(SARIMAXResultsWrapper, SARIMAXResults)  # noqa:E305

def show():
    print("Hello World!")