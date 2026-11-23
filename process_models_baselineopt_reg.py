"""
Baseline-opt FOPDT model with regularized objective.

Regularization pulls tau and theta toward physiological priors and y0 toward
the pre-window median.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import stats


class Data:
  def __init__(self, ts, y, u):
    self.ts = ts
    self.y = y
    self.u = u

  def __iter__(self):
    return iter([self.ts, self.y, self.u])


class Result:
  def __init__(self, p_opt, p_cov, p_val, r_square, y_hat, resid, RMSE):
    self.p_opt = p_opt
    self.p_cov = p_cov
    self.p_val = p_val
    self.r_square = r_square
    self.y_hat = y_hat
    self.resid = resid
    self.RMSE = RMSE


class ProcessModel:
  def __init__(self, t, y, u):
    self.Data = Data(t, y, u)
    self.uf = interp1d(t, u)

  def objective(self, params):
    ym = self.simulate(params)
    return np.sum((ym - self.Data.y) ** 2)

  def _fit_params(self, init_guess=None):
    if init_guess is None:
      init_guess = np.ones(self.params.shape)
    solution = minimize(self.objective, init_guess, bounds=self.bounds)
    return solution.x

  def fit_model(self, plot_result=False, plot_diagnostics=False):
    p_opt = self._fit_params()
    y_hat = self.simulate(p_opt, self.Data.ts, self.Data.u)
    resid = self.Data.y - y_hat
    rmse = np.sqrt(np.sum(resid ** 2) / len(self.Data.y))
    r_square = 1 - (np.var(self.Data.y - y_hat) / np.var(self.Data.y))
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
    Compute the analytic Jacobian for the FOPDT model parameters (K, tau, theta, y0).
    """
    K, tau, theta, y0 = params
    uf = interp1d(self.Data.ts, self.Data.u, fill_value="extrapolate")
    J = np.zeros((len(self.Data.ts), 4))
    dt = self.Data.ts[1] - self.Data.ts[0]
    for i, t in enumerate(self.Data.ts):
      if t - theta <= 0:
        u_delayed = 0
        u_derivative = 0
      else:
        u_delayed = uf(t - theta)
        u_derivative = (uf(t - theta + dt) - uf(t - theta)) / dt
      J[i, 0] = u_delayed / tau
      J[i, 1] = -(K * u_delayed - (self.Data.y[i] - y0)) / tau ** 2
      J[i, 2] = -(K / tau) * u_derivative
      J[i, 3] = 1.0 / tau
    return J

  def estimate_covariance(self, params):
    res = self.result.resid
    dof = len(self.Data.ts) - len(params)
    if dof <= 0:
      print("Not enough degrees of freedom to compute confidence intervals.")
      return None

    sigma2 = np.sum(res ** 2) / dof
    J = self.jacobian(params)
    if not np.isfinite(sigma2) or not np.isfinite(J).all():
      print("Non-finite values in covariance calculation. Cannot compute confidence intervals.")
      return None

    try:
      C = sigma2 * np.linalg.pinv(J.T @ J)
    except (np.linalg.LinAlgError, ValueError):
      print("Jacobian is singular or invalid. Cannot compute confidence intervals.")
      return None

    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.isfinite(C).all():
      print("Invalid covariance matrix. Cannot compute confidence intervals.")
      return None

    return C

  def pvalue(self, params, p_cov):
    if p_cov is None:
      return np.full(len(params), np.nan)

    p_cov = np.asarray(p_cov)
    if p_cov.ndim != 2 or p_cov.shape[0] != p_cov.shape[1]:
      return np.full(len(params), np.nan)

    se = np.sqrt(np.diag(p_cov))
    t_values = params / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), len(self.Data.ts) - len(params)))
    return p_values

  def plot_results(self, show=True):
    ts, ys, us = self.Data
    res = self.result
    sem = np.sqrt(np.sum((res.resid) ** 2) / (len(ys) - 3))
    pi95 = 1.96 * sem

    fig, ax1 = plt.subplots(figsize=(8, 6))
    title = f"Model fit: Rsq = {res.r_square:0.3f}, RMSE = {res.RMSE:0.3f}"
    if hasattr(self, "plot_context") and isinstance(self.plot_context, dict):
      ctx = self.plot_context
      ctx_text = (
        f"model={ctx.get('model')} | pre={ctx.get('pre')} | post={ctx.get('post')} | "
        f"lowcut={ctx.get('lowcut')} | highcut={ctx.get('highcut')}"
      )
      ax1.set_title(f"{title}\n{ctx_text}", fontsize=11)
    else:
      ax1.set_title(title)

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

    K, tau, theta, y0 = res.p_opt
    ax1.text(
      0.5,
      -0.15,
      f"K = {K:.3f}, tau = {tau:.3f}, theta = {theta:.3f}, y0 = {y0:.3f}",
      transform=ax1.transAxes,
      ha='center',
      va='top',
      fontsize=12,
    )

    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Change in input', color=color)
    ax2.plot(ts, us, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([min(us) - (0.1 * max(us)), 10 * max(us)])

    fig.tight_layout()
    if show:
      plt.show()
    return fig


class FOPDT(ProcessModel):
  """
  FOPDT with regularized objective:
    J = SSE + lam_tau*(tau-5)^2 + lam_theta*(theta-2)^2 + lam_y0*(y0-y0_ref)^2
  """
  def __init__(
      self,
      t,
      y,
      u,
      params=np.ones((4,)),
      pre=None,
      lam_tau=1.0,
      lam_theta=1.0,
      lam_y0=1.0,
      tau0=5.0,
      theta0=2.0,
  ):
    super().__init__(t, y, u)
    self.params = params
    # K and y0 are unrestricted; tau and theta use physiological time limits.
    self.bounds = [(-np.inf, np.inf), (0.1, 30.0), (0.0, 10.0), (-np.inf, np.inf)]
    self.tmin = min(t)
    self.lam_tau = lam_tau
    self.lam_theta = lam_theta
    self.lam_y0 = lam_y0
    self.tau0 = tau0
    self.theta0 = theta0
    if pre is not None:
      self.y0_ref = float(np.median(y[:pre]))
    else:
      self.y0_ref = float(np.median(y))

  def objective(self, params):
    K, tau, theta, y0 = params
    ym = self.simulate(params)
    sse = np.sum((ym - self.Data.y) ** 2)
    pen = (
      self.lam_tau * (tau - self.tau0) ** 2 +
      self.lam_theta * (theta - self.theta0) ** 2 +
      self.lam_y0 * (y0 - self.y0_ref) ** 2
    )
    return sse + pen

  def model(self, y, t, uf, K, tau, theta, y0):
    try:
      if t - theta <= self.tmin:
        um = 0
      else:
        um = uf(t - theta)
    except:
      um = 0
    dydt = (-(y - y0) + K * (um)) / tau
    return dydt

  def simulate(self, params=None, t=None, u=None):
    if params is None:
      params = self.params
    K, tau, theta, y0 = params

    if t is None:
      t = self.Data.ts
    if u is None:
      u = self.Data.u

    uf = interp1d(t, u)
    ym = np.zeros(len(t))
    ym[0] = y0

    for i in range(len(t) - 1):
      ts = [t[i], t[i + 1]]
      y1 = odeint(self.model, ym[i], ts, args=(uf, K, tau, theta, y0))[-1].item()
      ym[i + 1] = y1
    return ym
