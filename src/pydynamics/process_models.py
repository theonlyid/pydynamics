"""
Module containing classes for process models and fitting dynamic models to data.

Author: Ali Zaidi

Version: 0.0.1

(C) 2025 Ali Zaidi. All rights reserved.
"""


import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import stats

import pickle

class Data:
  """
  A data container for time series data.

  Attributes:
    ts (array-like): Array of time points.
    y (array-like):  Array of output data corresponding to the time points.
    u (array-like):  Array of input data corresponding to the time points.
  """
  def __init__(self, ts, y, u):
    self.ts = ts
    self.y = y
    self.u = u

  def __iter__(self):
    return iter([self.ts, self.y, self.u])

  def __repr__(self):
    return f"Timeseries data of length {len(self.ts)}"
  
  def __str__(self):
    return f"Timeseries data of length {len(self.ts)}"
  

class Result:
  """
  A result container for process model fitting. Contains the optimized parameters, covariance, p-values, R-squared value, model predictions, residuals, and RMSE.
  
  Attributes:
    p_opt (array-like): Optimized parameters.
    p_cov (array-like): Covariance matrix of the parameters.
    p_val (array-like): P-values of the parameters.
    r_square (float): R-squared value of the model fit.
    y_hat (array-like): Model predictions.
    resid (array-like): Residuals of the model fit.
    RMSE (float): Root Mean Squared Error of the model fit.
  """
  def __init__(self, p_opt, p_cov, p_val, r_square, y_hat, resid, RMSE):
    self.p_opt = p_opt
    self.p_cov = p_cov
    self.p_val = p_val
    self.r_square = r_square
    self.y_hat = y_hat
    self.resid = resid
    self.RMSE = RMSE

  def __repr__(self):
    return f"Result(p_opt={self.p_opt}, p_val={self.p_val}, r_square={self.r_square}, RMSE={self.RMSE})"

  def __str__(self):
    return f"Result(p_opt={self.p_opt}, p_val={self.p_val}, r_square={self.r_square}, RMSE={self.RMSE})"


class ProcessModel:
  """
  Base model class that handles the fitting and plotting for various process models.

  Attributes:
    Data (Data): An instance of the Data class containing time (t), output (y), and input (u) data.
    uf (interp1d): Interpolated function of the input data.
  """

  def __init__(self, t, y, u):
    """
    Initialize the process model with time, output, and input data.

    Args:
      t (array-like): Array of time points.
      y (array-like): Array of output data corresponding to the time points.
      u (array-like): Array of input data corresponding to the time points.
    """
    self.Data = Data(t, y, u)
    self.uf = interp1d(t, u)

  def objective(self, params):
    """
    Calculate the objective function value for given parameters.

    This function simulates the model with the provided parameters and computes
    the sum of squared differences between the simulated output and the actual data.

    Args:
      params (array-like): The parameters to be used for simulation.

    Returns:
      float: The sum of squared differences between the simulated output and the actual data.
    """
    ym = self.simulate(params)
    return np.sum((ym - self.Data.y) ** 2)

  def _fit_params(self, init_guess=None):
    """
    Fit the parameters of the model using optimization.

    This is an internal method and shouldn't be used directly. Use `fit_model` instead.

    Args:
      init_guess (array-like, optional): Initial guess for the parameters. If None, an array of ones with the same shape as `self.params` is used.

    Returns:
      p_opt (array-like): Optimized parameters.
    """
    if init_guess is None:
      init_guess = np.ones(self.params.shape)
    solution = minimize(self.objective, init_guess, bounds=self.bounds)
    p_opt = solution.x
    return p_opt

  def fit_model(self, plot_result=False, plot_diagnostics=False):
    """
    Fits the model to the data, computes residuals, RMSE, R-squared value, and parameter statistics.

    Args:
      plot_result (bool, optional): If True, plots the model results. Default is False.
      plot_diagnostics (bool, optional): If True, plots diagnostic information. Default is False.

    Returns:
      Result: An object containing the fitted parameters, their covariance, p-values, R-squared value, 
      model predictions, residuals, and RMSE.
    """
    p_opt = self._fit_params()  # fit the model
    y_hat = self.simulate(p_opt, self.Data.ts, self.Data.u)  # generate model predictions
    resid = self.Data.y - y_hat  # compute residuals
    rmse = np.sqrt(np.sum(resid ** 2) / len(self.Data.y))  # compute RMSE
    r_square = 1 - (np.var(self.Data.y - y_hat) / np.var(self.Data.y))  # compute R-squared value
    self.result = Result(p_opt=p_opt, p_cov=None, p_val=None, r_square=r_square, y_hat=y_hat, resid=resid, RMSE=rmse)
    p_cov = self.estimate_covariance(p_opt)
    p_val = self.pvalue(p_opt, p_cov)
    self.result = Result(p_opt=p_opt, p_cov=p_cov, p_val=p_val, r_square=r_square, y_hat=y_hat, resid=resid, RMSE=rmse)
    ks = stats.kstest(resid, 'norm', (0, resid.std()))
    print(f"KS Test of Residuals: p={ks.pvalue:0.3f}")

    if plot_result:
      self.plot_results()

    if plot_diagnostics:
      self.diagnostics()
    return self.result

  def jacobian(self, params) -> np.ndarray:
    """
    Compute the Jacobian matrix of the model with respect to the parameters.

    Args:
      params (array-like): The parameters for which the Jacobian is to be computed.

    Returns:
      np.ndarray: The Jacobian matrix.
    """
    K, tau, theta = params
    uf = interp1d(self.Data.ts, self.Data.u, fill_value="extrapolate")
    J = np.zeros((len(self.Data.ts), 3))
    dt = self.Data.ts[1] - self.Data.ts[0]

    for i, t in enumerate(self.Data.ts):
      if t - theta <= 0:
        u_delayed = 0
        u_derivative = 0
      else:
        u_delayed = uf(t - theta)
        u_derivative = (uf(t - theta + dt) - uf(t - theta)) / dt

      J[i, 0] = u_delayed / tau
      J[i, 1] = -(K * u_delayed - self.Data.y[i]) / tau ** 2
      J[i, 2] = -(K / tau) * u_derivative

    return J

  def estimate_covariance(self, params):
    """
    Estimate the covariance matrix of the parameters.

    Args:
      params (array-like): The parameters for which the covariance matrix is to be estimated.

    Returns:
      np.ndarray or None: The estimated covariance matrix of the parameters. If the Jacobian is singular,
      returns None and prints an error message.
    """
    res = self.result.resid
    sigma2 = np.sum(res ** 2) / (len(self.Data.ts) - len(params))
    J = self.jacobian(params)

    # Covariance matrix
    try:
      C = sigma2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
      print("Jacobian is singular. Cannot compute confidence intervals.")
      return None

    return C

  def estimate_confidence_intervals(self, p_cov):
    """
    Compute 95% confidence intervals of MLE parameters from the covariance matrix.

    Args:
      p_cov (np.ndarray): Covariance matrix of the parameters.

    Returns:
      list: List of tuples containing the lower and upper bounds of the confidence intervals for each parameter.
    """
    se = np.sqrt(np.diag(p_cov))
    p_opt = self.result.p_opt

    # 95% Confidence intervals
    confidence_intervals = []

    for i in range(len(p_opt)):
      lower = p_opt[i] - 1.96 * se[i]
      upper = p_opt[i] + 1.96 * se[i]
      confidence_intervals.append((lower, upper))
    return confidence_intervals

  def pvalue(self, params, p_cov):
    """
    Compute the p-values for the model parameters.

    Args:
      params (np.ndarray): Model parameters.
      p_cov (np.ndarray): Covariance matrix of the parameters.

    Returns:
      np.ndarray: Array of p-values for each parameter.
    """
    se = np.sqrt(np.diag(p_cov))
    t_values = params / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), len(self.Data.ts) - len(params)))
    return p_values

  @staticmethod
  def ttest(mu1, sigma1, n1, mu2, sigma2, n2) -> float:
    """
    Perform a two-sample t-test for the difference between two means.

    To be used to compare a parameter estimate from two different models / datasets (e.g. c_toi vs r_toi).

    Args:
      mu1 (float): Mean of the first sample.
      sigma1 (float): Standard deviation of the first sample.
      n1 (int): Number of datapoints from the first sample.
      mu2 (float): Mean of the second sample.
      sigma2 (float): Standard deviation of the second sample.
      n2 (int): Number of datapoints for the second sample.

    Returns:
      t-stat (float): The calculated t-value for the two-sample t-test.
      p-value (float): The calculated p-value for the two-sample t-test.
    """
    t_value = (mu1 - mu2) / np.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), n1 + n2 - 2))

    return t_value, p_value

  def save_model(self, filename):
    """
    Save the model object to a file using pickle.

    Args:
      filename (str): The name of the file to save the model object to.

    Returns:
      None
    """
    with open(filename, 'wb') as file:
      pickle.dump(self, file)

  def plot_confidence_intervals(self):
    """
    Plots the observed data, the best fit model, and the 95% confidence intervals for the model predictions.

    This method uses the time series data, the observed response, and the predicted response from the model.
    It calculates the confidence intervals based on the covariance of the parameter estimates and simulates
    the response using the lower and upper bounds of the confidence intervals. The plot includes the observed
    data points, the best fit model line, and a shaded area representing the 95% confidence interval.
    """
    t = self.Data.ts
    u = self.Data.u
    y_true = self.Data.y  # Observed data
    y_pred = self.result.y_hat
    params = self.result.p_opt
    ci = self.ci_from_cov(self.result.p_cov)

    # Simulate using confidence interval bounds
    params_lower = [ci[i][0] for i in range(len(params))]
    params_upper = [ci[i][1] for i in range(len(params))]
    y_lower = self.simulate(params_lower, self.Data.ts, self.Data.u)
    y_upper = self.simulate(params_upper, self.Data.ts, self.Data.u)

    plt.figure(figsize=(8, 5))
    plt.plot(t, y_true, 'ko', markersize=3, label="Observed Data")
    plt.plot(t, y_pred, 'r-', label="Best Fit Model")
    plt.fill_between(t, y_lower, y_upper, color='red', alpha=0.3, label="95% Confidence Interval")

    plt.xlabel("Time (s)")
    plt.ylabel("Response y(t)")
    plt.title("Model Fit with Confidence Interval")
    plt.legend()
    plt.grid(True)
    plt.show()

  def step(self, params=None, step=1):
    """
    Generate a step response plot for a First Order Plus Dead Time (FOPDT) model with the given parameters.

    Args:
      params (list or None): Parameters for the FOPDT model. If None, uses self.result.p_opt.
      step (int): The step input value. Default is 1.

    Returns:
      None: This function generates and displays a plot.

    The function performs the following steps:
    1. If params is None, it assigns self.result.p_opt to params.
    2. Extracts the time constant (taum) from params.
    3. Creates a time array (ts) ranging from -taum to 5 * taum.
    4. Initializes an input array (us) with zeros and sets the step input after taum.
    5. Simulates the FOPDT model response using the given parameters, time array, and input array.
    6. Plots the output response and input signal on two subplots.
    """
    if params is None:
      params = self.result.p_opt
    taum = params[1]
    ts = np.arange(-taum, 5 * taum)

    us = np.zeros((len(ts)))
    us[np.int8(taum) + 1:] = step

    y = self.simulate(params, ts, us)
    plt.figure()
    plt.subplots(2, 1, sharex=True)

    plt.subplot(211)
    plt.plot(ts, y)
    plt.grid()
    plt.ylabel("Output")
    plt.subplot(212)
    plt.ylabel("Input")
    plt.plot(ts, us)
    plt.grid()

  def plot_results(self):
    """
    Plots the results of the model fit, including the observed data, model predictions, 
    and prediction intervals.

    This method generates a plot with two y-axes:
    - The left y-axis shows the change in output with observed data points, model predictions, 
      and prediction intervals.
    - The right y-axis shows the change in input over time.

    The plot includes:
    - A title displaying the R-squared value and RMSE of the model fit.
    - Observed data points as blue dots.
    - Model predictions as a black line.
    - 95% prediction intervals as a shaded red area.
    - Standard error of the mean (SEM) intervals as a shaded red area with higher opacity.
    - Change in input as a gray line (right axis).

    The method also adjusts the layout to ensure the right y-label is not clipped.
    """
    # gather variables
    ts, ys, us = self.Data
    res = self.result

    # calculate prediction intervals
    sem = np.sqrt(np.sum((res.resid) ** 2) / (len(ys) - 3))
    pi95 = 1.96 * sem

    fig, ax1 = plt.subplots(figsize=(8, 6))
    plt.title(f"Model fit: Rsq = {res.r_square:0.3f}, RMSE = {res.RMSE:0.3f}")

    color = 'tab:blue'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Change in output', color=color)
    ax1.fill_between(ts, res.y_hat - pi95, res.y_hat + pi95, color='r', alpha=0.15, label='95% PI')
    ax1.fill_between(ts, res.y_hat - sem, res.y_hat + sem, color='r', alpha=0.2, label='SE PI')
    ax1.plot(ts, ys, 'b.', label='observations', alpha=0.6)
    ax1.plot(ts, res.y_hat, 'k', linewidth=1.5, label='model')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:gray'
    ax2.set_ylabel('Change in input', color=color)  # we already handled the x-label with ax1
    ax2.plot(ts, us, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([min(us) - (0.1 * max(us)), 10 * max(us)])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

  def diagnostics(self, plot=True):
    """
    Perform diagnostics on the model's residuals and input data.

    This method calculates and prints the mean squared values of the 
    autocorrelation and cross-correlation functions of the residuals 
    and their differences. Optionally, it plots these diagnostics.

    Args:
      plot (bool, optional): If True, plots the diagnostics. Default is True.

    Attributes:
      cc_means (dict): A dictionary containing the mean squared values of the 
      autocorrelation and cross-correlation functions:
        - "acf": Autocorrelation of residuals
        - "ccf": Cross-correlation of residuals with input
        - "acfd": Autocorrelation of residual differences
        - "ccfd": Cross-correlation of residual differences with input

    Prints:
      Mean absolute values of the autocorrelation and cross-correlation 
      functions.

    Plots:
      If plot is True, generates the following plots:
        - Residuals
        - Autocorrelation of residuals
        - Cross-correlation of residuals with input
        - Residual differences
        - Autocorrelation of residual differences
        - Cross-correlation of residual differences with input
    """
    t = self.Data.ts
    dt = t[1] - t[0]
    resid = self.result.resid
    res_norm = resid / np.linalg.norm(resid)
    u_norm = self.Data.u / np.linalg.norm(self.Data.u)

    resd = np.diff(resid)
    resd_norm = np.diff(resd) / np.linalg.norm(np.diff(resd))

    acf = np.correlate(res_norm, res_norm, 'same')
    acf[len(acf) // 2] = 0
    ccf = np.correlate(u_norm, res_norm, 'same')

    acfd = np.correlate(resd_norm, resd_norm, 'same')
    acfd[len(acfd) // 2] = 0
    ccfd = np.correlate(u_norm, resd_norm, 'same')

    self.cc_means = dict({"acf": np.mean(acf ** 2), "ccf": np.mean(ccf ** 2), "acfd": np.mean(acfd ** 2), "ccfd": np.mean(ccfd ** 2)})

    print(f"Residual ACF:{np.mean(acf ** 2):0.3f}, CCF:{np.mean(ccf ** 2):0.3f}, ACFD:{np.mean(acfd ** 2):0.3f}, CCFD:{np.mean(ccfd ** 2):0.3f}")

    if plot:
      plt.subplots(3, 2)
      plt.subplot(321)
      plt.plot(resid)
      plt.title("Residuals")
      plt.grid()

      plt.subplot(323)
      plt.title("Autocorrelation of Residuals")
      plt.plot(acf)
      plt.ylim([-1, 1])
      plt.grid()

      plt.subplot(325)
      plt.title("Cross-correlation: Residuals -> input")
      plt.plot(ccf)

      plt.ylim([-1, 1])
      plt.grid()

      plt.subplot(322)
      plt.plot(resd)
      plt.title("Residual diff")
      plt.grid()

      plt.subplot(324)
      plt.title("Autocorrelation of Residual diff")
      plt.plot(acfd)
      plt.ylim([-1, 1])
      plt.grid()

      plt.subplot(326)
      plt.title("Cross-correlation: Residual diff -> input")
      plt.plot(ccfd)
      plt.ylim([-1, 1])
      plt.grid()


class FOPDT(ProcessModel):
  """
  First-Order Plus Dead Time (FOPDT) process model.
  This class represents a first-order plus dead time (FOPDT) process model, which is commonly used in process control to describe the dynamic behavior of a system.

  Attributes:
    params (numpy.ndarray): An array containing the default parameters of the system (K, tau, theta).
    bounds (list): A list of tuples specifying the bounds for the parameters (K, tau, theta).
    tmin (float):  The minimum time value in the time vector `t`.

  """
  def __init__(self, t, y, u, params=np.ones((3,))):
    """
    Initialize the FOPDT model with time, output, and input data.
    
    Args:
      t (array-like): Array of time points.
      y (array-like): Array of output data corresponding to the time points.
      u (array-like): Array of input data corresponding to the time points.
      params (array-like, optional): Default parameters for the FOPDT model. Default is np.ones((3,)).
    """
    super(FOPDT, self).__init__(t, y, u)
    self.params = params
    self.bounds = [(-np.inf, np.inf), (0.1, np.inf), (0., np.inf)]
    self.tmin = min(t)

  def model(self, y, t, uf, K, tau, theta):
    """
    Computes the derivative of the system state `y` at time `t` for a given input function `uf`.

    Args:
      y (float): The current state of the system.
      t (float): The current time.
      uf (function): A function representing the input to the system, which takes time `t` as an argument.
      K (float): The system gain.
      tau (float): The system time constant.
      theta (float): The time delay of the system.

    Returns:
      dy/dt (float): The derivative of the system state `y` at time `t`.
    """
    try:
      if t - theta <= self.tmin:
        um = 0
      else:
        um = uf(t - theta)
    except:
      um = 0
    dydt = (-(y) + K * (um)) / tau
    return dydt

  def simulate(self, params=None, t=None, u=None):
    """
    Simulate the dynamic system using the provided parameters, time vector, and input signal.

    Args:
      params (tuple, optional): A tuple containing the system parameters (K, tau, theta). If not provided, the default parameters of the object are used.
      t (array-like, optional): The time vector for the simulation. If not provided, the default time vector of the object is used.
      u (array-like, optional): The input signal for the simulation. If not provided, the default input signal of the object is used.

    Returns:
      ym (numpy.ndarray): The simulated output of the dynamic system over the given time vector.
    """
    if params is None:
      params = self.params
    K, tau, theta = params

    if t is None:
      t = self.Data.ts

    if u is None:
      u = self.Data.u

    uf = interp1d(t, u)
    ym = np.zeros(len(t))

    for i in range(len(t) - 1):
      ts = [t[i], t[i + 1]]
      y1 = odeint(self.model, ym[i], ts, args=(uf, K, tau, theta))[-1].item()
      ym[i + 1] = y1

    return ym


class SOPDT(ProcessModel):
  def __init__(self, t, y, u, params=np.ones((4,))):
    super(SOPDT, self).__init__(t, y, u)
    self.params = params
    self.bounds = [(-np.inf, np.inf), (0.01, np.inf), (0.01, np.inf), (0, np.inf)]

  def model(self, x, t, uf, Kp, taus, zeta, thetap):
    
    # Kp = process gain
    # taus = second order time constant
    # zeta = damping factor
    # thetap = model time delay
    # equation: ts^2 dy2/dt2 + 2 zeta taus dydt + y = Kp * u(t-thetap)

    u0 = 0
    try:
      if (t - thetap) <= 0:
        um = 0
      else:
        um = uf(t - thetap)
    except:
      # catch any error
      um = 0
    # two states (y and y')
    y = x[0]
    dydt = x[1]
    dy2dt2 = (-2.0 * zeta * taus * dydt - y + Kp * (um)) / taus ** 2
    return [dydt, dy2dt2]

  # simulate model with x=[Km, taum, zetam, thetam]
  def simulate(self, params=None):
    """
    Simulate the system using the given parameters.

    Parameters:
    params (tuple, optional): A tuple containing the parameters (Kp, taus, zeta, thetap) 
                  for the simulation. If None, self.params will be used.
    
    Returns:
    numpy.ndarray: The simulated output values over time.
    """
    if params is None:
      params = self.params
    Kp, taus, zeta, thetap = params

    uf = self.uf
    t = self.Data.t
    # storage for model values
    xm = np.zeros((len(t), 2))  # model
    # initial condition
    xm[0] = 0
    # loop through time steps
    for i in range(len(t) - 1):
      ts = [t[i], t[i + 1]]
      inputs = (uf, Kp, taus, zeta, thetap)
      # integrate SOPDT model
      x = odeint(self.model, xm[i], ts, args=inputs)
      xm[i + 1] = x[-1]
    y = xm[:, 0]
    return y


if __name__ == '__main__':
  
  # Generate sample data
  t, y, u  = np.ones((3,100))
  t = np.cumsum(t)
  u[:5] = 0
  u[60:] = 0

  # Test First Order Model
  fom = FOPDT(t, y, u) # initialize the model
  ys = fom.simulate(np.array([2., 5., 5.])) # simulate the model with given parameters
  noise = np.random.normal(0, 0.5, len(ys)) # add noise to the output
  yn = ys + noise
  fom = FOPDT(t, yn, u) # initialize the model with noisy data
  result = fom.fit_model(plot_result=True) # fit the model to the data and plot the results
  print(f"Params: K: {result.p_opt[0]:0.3f}, tau: {result.p_opt[1]:0.3f}, theta: {result.p_opt[2]:0.3f}")
  print(f"P-values: K: {result.p_val[0]:0.3f}, tau: {result.p_val[1]:0.3f}, theta: {result.p_val[2]:0.3f}")
