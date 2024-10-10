import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class FOPDT():
  def __init__(self, t, y, u):
    self.t = t
    self.y = y
    self.u = u
    self.uf = interp1d(t, u)
    self.bounds = [(-np.inf, np.inf), (0.1, np.inf), (0., np.inf)]

    
  def model(self, y, t, uf, K, tau, theta):
    try:    
      if t - theta <= 0:
        um = 0
      else:
        um = uf(t - theta)
    except:
      um = 0
    dydt = (-(y) + K*(um))/tau
    return dydt

  def simulate(self, params=None, t=None, u=None):
    if params is None:
      params=self.params
    Km = params[0]
    taum = params[1]
    thetam = params[2]

    if t is None:
      t = self.t
    
    if u is None:
      u = self.u
    
    uf = interp1d(t, u)
    ym = np.zeros(len(t))

    for i in range(0, len(t)-1):
      ts=[t[i], t[i+1]]
      y1 = odeint(self.model, ym[i], ts, args=(uf, Km, taum, thetam))[-1].item()
      ym[i+1] = y1

    return ym

  def objective(self, params):
    ym = self.simulate(params)
    return np.sum((ym -self.y)**2)

  def fit_params(self, init_guess=None):
    if init_guess is None:
      init_guess = np.ones((3,))
    solution = minimize(self.objective, init_guess, bounds=self.bounds)
    self.params_opt = solution.x
    return self.params_opt
  
  def step(self, params):
    """
    Generate a step response plot for a FOPDT model with the given params
    """
    params = self.params_opt
    Km = params[0]
    taum = params[1]
    thetam = params[2]
    ts = np.arange(-taum, 5*taum)

    us = np.zeros((len(ts)))
    us[np.int8(taum)+1:] = 1

    y = self.simulate(params, ts, us)
    plt.figure()
    plt.subplots(2,1, sharex=True)

    plt.subplot(211)
    plt.plot(ts, y)
    plt.grid()
    plt.ylabel("Output")
    plt.subplot(212)
    plt.plot(ts, us)
    plt.grid()


  def fit_model(self, plot_result=False):
    result = dict()
    result["soln"] = self.fit_params()
    result["y_pred"] = self.simulate(self.params_opt)
    result["mse"] = self.objective(self.params_opt)/len(self.y)
    result["R2"] = 1 - (np.var(self.y - result["y_pred"])/np.var(self.y))
    self.result = result
  
    if plot_result:

      plt.figure()
      plt.subplots(2,1, sharex=True)
      plt.suptitle("model fit")
      
      plt.subplot(211)
      plt.plot(self.t, self.y, linewidth=2)
      plt.plot(self.t, result["y_pred"], 'k--')
      plt.grid()
      plt.title("Output")
      plt.ylabel("Change in output")
      plt.legend(["data", "model"])

      plt.subplot(212)
      plt.plot(self.t, self.u)
      plt.grid()
      plt.title("Input")
      plt.xlabel("time")
      plt.ylabel("change in input")

    return result
    