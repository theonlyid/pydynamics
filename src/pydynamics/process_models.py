import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d

#TODO: Generate Results class with named tuples

Data = namedtuple("Data", ["ts", "y", "u"])
Params = namedtuple("Params", ["K", "tau", "theta", "zeta"])
Result = namedtuple("Result", ["p_opt", "p_cov", "r_square", "y_hat", "resid"])

class ProcessModel:
  "Base model class that handles the fitting and plotting for various process models"

  def __init__(self, t, y, u):
    self.t = t
    self.y = y
    self.u = u
    self.uf = interp1d(t, u)

  def objective(self, params):
    ym = self.simulate(params)
    return np.sum((ym -self.y)**2)

  def fit_params(self, init_guess=None):
    if init_guess is None:
      init_guess = np.ones(self.params.shape)
    solution = minimize(self.objective, init_guess, bounds=self.bounds)
    p_opt = solution.x
    return p_opt
  
  def fit_model(self, plot_result=False):
    result = dict()
    result["soln"] = self.fit_params()
    result["y_pred"] = self.simulate(result["soln"])
    result["mse"] = self.objective(result["soln"])/len(self.y)
    result["R2"] = 1 - (np.var(self.y - result["y_pred"])/np.var(self.y))
    self.result = result
  
    if plot_result:
      self.plot_results()

    self.result = result
    return result

  def step(self, params=None, step=1):
    """
    Generate a step response plot for a FOPDT model with the given params
    """
    if params is None:
      params = self.result["soln"]
    taum = params[1]
    ts = np.arange(-taum, 5*taum)

    us = np.zeros((len(ts)))
    us[np.int8(taum)+1:] = step

    y = self.simulate(params, ts, us)
    plt.figure()
    plt.subplots(2,1, sharex=True)

    plt.subplot(211)
    plt.plot(ts, y)
    plt.grid()
    plt.ylabel("Output")
    plt.subplot(212)
    plt.ylabel("Input")
    plt.plot(ts, us)
    plt.grid()

  def plot_results(self):
    plt.subplots(2,1, sharex=True)
    plt.suptitle("model fit")
    
    plt.subplot(211)
    plt.plot(self.t, self.y, linewidth=2)
    plt.plot(self.t, self.result["y_pred"], 'k--')
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

  def diagnostics(self, resid, plot=True):

    t = self.t
    dt = t[1]-t[0]
   
    res_norm = resid/np.linalg.norm(resid)
    u_norm = self.u/np.linalg.norm(self.u)

    resd = np.diff(resid)
    resd_norm = np.diff(resd)/np.linalg.norm(np.diff(resd))

    acf = np.correlate(res_norm, res_norm, 'same')
    acf[len(acf)//2] = 0
    ccf = np.correlate(u_norm, res_norm, 'same')

    acfd = np.correlate(resd_norm, resd_norm, 'same')
    acfd[len(acfd)//2] = 0
    ccfd = np.correlate(u_norm, resd_norm, 'same')

    self.cc_means = dict({"acf": np.mean(acf**2), "ccf": np.mean(ccf**2), "acfd": np.mean(acfd**2), "ccfd": np.mean(ccfd**2)})

    print(f"{np.mean(acf**2):0.3f}, {np.mean(ccf**2):0.3f}, {np.mean(acfd**2):0.3f}, {np.mean(ccfd**2):0.3f}")


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

      plt.ylim([-1,1])
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
      plt.ylim([-1,1])
      plt.grid()


class FOPDT(ProcessModel):
  def __init__(self, t, y, u, params=np.ones((3,))):
    super(FOPDT, self).__init__(t, y, u)
    self.params = params
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
            if (t-thetap) <= 0:
                um = 0
            else:
                um = uf(t-thetap)
        except:
            # catch any error
            um = 0
        # two states (y and y')
        y = x[0]
        dydt = x[1]
        dy2dt2 = (-2.0*zeta*taus*dydt - y + Kp*(um))/taus**2
        return [dydt,dy2dt2]

    # simulate model with x=[Km, taum, zetam, thetam]
    def simulate(self, params=None):
        if params is None:
            params = self.params

        # input arguments
        Kp = params[0]
        taus = params[1]
        zeta = params[2]
        thetap = params[3]
        
        uf = self.uf
        t = self.t
        # storage for model values
        xm = np.zeros((len(t),2))  # model
        # initial condition
        xm[0] = 0
        # loop through time steps    
        for i in range(0, len(t)-1):
            ts = [t[i],t[i+1]]
            inputs = (uf, Kp, taus, zeta, thetap)
            # integrate SOPDT model
            x = odeint(self.model, xm[i], ts, args=inputs)
            xm[i+1] = x[-1]
        y = xm[:,0]
        return y


if __name__ == '__main__':
  t, y, u  = np.ones((3,100))
  t = np.cumsum(t)
  u[:5] = 0
  u[60:] = 0

  # Simulate data with second order model
  som = SOPDT(t, y, u)
  ys = som.simulate(np.array([2., 5., 0.3, 5.]))
  som.y = ys

  # Fit Second Order Model to simulated data
  result = som.fit_model(plot_result=False)
  print(result["soln"])

  # Test First Order Model
  fom = FOPDT(t, ys, u)
  result = fom.fit_model(plot_result=True)
  resid = fom.y - result["y_pred"]
  fom.diagnostics(resid)
  plt.show()
