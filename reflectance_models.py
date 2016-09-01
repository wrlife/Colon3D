# Author: True Price <jtprice at cs.unc.edu>


import numpy as np
import skimage.color
import util

from itertools import izip
from scipy.optimize import minimize

import sfs # for *_MODEL enum mappings

from sklearn import linear_model

MERL_RED_SCALE = 1. / 1500.
MERL_GREEN_SCALE = 1.15 / 1500.
MERL_BLUE_SCALE = 1.66 / 1500.

# load an empirical BRDF from the MERL database;
# since we're only interested in a colinear light/angle assumption, the BRDF
# returned only indexes those values
# returns:
#   BRDF: 3xN numpy array of BRDF values, where each row is a different channel,
#         and the columns correspond to different angles between the normal and
#         light directions (a nonlinear mapping; see merl_angle_to_index)
def load_merl_brdf(brdf_file):
  # read in brdf
  with open(brdf_file, 'rb') as f:
    dims = np.fromfile(f, dtype=np.int32, count=3)
    brdf = np.fromfile(f, dtype=np.float64)

  # we only care about the case where the light direction is colinear with the
  # camera
  brdf = brdf.reshape((3,) + tuple(dims))[:,:,0,0].copy()
  brdf[0] *= MERL_RED_SCALE
  brdf[1] *= MERL_GREEN_SCALE
  brdf[2] *= MERL_BLUE_SCALE

  return brdf

# converts an angle (in radians) to an index in a MERL BRDF array:
#   convert to an angle in degrees, then multiply by 90, then take the sqrt,
#   then convert to an integer; this is the corresponding index in the BRDF
def merl_angle_to_index(theta):
  return np.sqrt(90. * np.degrees(theta)).astype(np.uint8)

# converts MERL BRDF angles to a corresponding index
#   index = int(sqrt(90 * deg(theta)))
def merl_index_to_angle(idx):
  return np.radians(idx * idx * merl_index_to_angle._CONST)
merl_index_to_angle._CONST = 1. / 90. # to avoid costly division each time

#
# Reflectance model fitting functions
# General parameters for the functions are:
#
# L: length-N array of image intensities in [0, 1]
# r: length-N array of depth values
# ndotl: length-N array of cos(theta) values
# falloff: r^m falloff coefficient for the light source
#

# Lambertian BRDF
def fit_lambertian(L, r, ndotl,width,height, falloff=2.):
  ndotl_over_r = ndotl * np.power(r, -falloff)
  alpha = np.sum(L * ndotl_over_r) / np.sum(ndotl_over_r * ndotl_over_r)

  residual = np.sum((L - alpha * ndotl_over_r)**2)

  import pdb;pdb.set_trace()
  T=alpha * ndotl_over_r
  T=T/np.amax(T)
  skimage.io.imsave('test.png',T.reshape(height,width))
  return np.array([alpha]), residual

# Oren-Nayar BRDF
def fit_oren_nayar(L, r, ndotl, falloff=2.):
  A = np.hstack((ndotl[:,np.newaxis], (1. - ndotl * ndotl)[:,np.newaxis]))
  A *= np.power(r, -falloff)[:,np.newaxis]

  coeff, residual, _, _ = np.linalg.lstsq(A, L)

  return coeff, residual[0]

# Oren-Nayar BRDF
# variables to solve for:
#   0: scaling coefficient (related to albedo and light intensity)
#   1: sigma^2 term
#def fit_oren_nayar(L, r, ndotl, falloff=2.):
#  sin_sq_term = (1. - ndotl * ndotl)
#  r_term = np.power(r, -falloff)
#
#  def f(X):
#    f.est = r_term * X[0] * (
#        (1. - 0.5 * (X[1] / (X[1] + 0.57))) * ndotl +
#        0.45 * X[1] / (X[1] + 0.09) * sin_sq_term)
#
#    #f.est = np.minimum(f.est, 1.)
#
#    return np.sum((L - f.est)**2)
#
#  init_params = np.ones(2)
#  init_params[0] = 1.
#  init_params[1] = 45.
#
#  bounds = [(None, None)] * len(init_params)
#  bounds[0] = (0., None)
#  bounds[1] = (0., None)
#
#  result = minimize(f, init_params, method='L-BFGS-B', bounds=bounds,
#                    options={'disp': False, 'iprint': -1})
#
#  return result.x, result.fun

# Phong BRDF
# variables to solve for:
#   0: coefficient of diffuse term
#   1: coefficient of specular term
#   2: power on specular term
def fit_phong(L, r, ndotl, falloff=2.):
  r_term = np.power(r, -falloff)

  def f(X):
    f.est = r_term * (X[0] * ndotl + X[1] * np.power(ndotl, X[2]))

    #f.est = np.minimum(f.est, 1.)

    return np.sum((L - f.est)**2)

  #lambertian_alpha = fit_lambertian(L, r, ndotl)[0]
  init_params = np.ones(3)
  init_params[0] = 1.#0.5 * lambertian_alpha
  init_params[1] = 1.#0.5 * lambertian_alpha
  init_params[2] = 100.

  bounds = [(None, None)] * len(init_params)
  bounds[0] = (0., None)
  bounds[1] = (0., None)
  bounds[2] = (2., None)

  result = minimize(f, init_params, method='L-BFGS-B', bounds=bounds,
                    options={'disp': False, 'iprint': -1})

  return result.x, result.fun

# Powers-of-cosine BRDF model
# Additional parameters:
#   K: K in the reflectance model equation; total number of coefficients is 2K
#   alpha: the weight to give to the regularization of the coefficients; if we
#          assume some average error in our fitted values (e.g., < 0.01 error in
#          luminance, which ranges from 0 to 1), the total residual is on the
#          order of this error squared, divided by 2
def fit_power_model(L, ori_r, r, ori_ndotl,ndotl,  width,height,K, falloff=2.0, alpha=0.5):
  sin_term = np.sqrt((1. - ndotl) * 0.5)

  # build a base-level least-squares coefficient matrix having N rows
  # and 2K columns; the first K columns correspond to the ndotl^k terms; the
  # last K columns correspond to the sin(theta/2) * ndotl^k terms
  A = np.empty((L.size, 2 * K))
  ndotlk = ndotl.copy()
  for k in xrange(K):
    A[:,k] = ndotlk
    A[:,k+K] = sin_term * ndotlk
    ndotlk *= ndotl

  coe=0.5/(1-min(r)/max(r))
  falloffs=coe*(-r/max(r)+1)+1.5
  A *= np.power(r, -falloffs)[:,np.newaxis]

  model = linear_model.Ridge(alpha=alpha, copy_X=True, fit_intercept=False)

  # solve for coefficients
  model.fit(A, L)
  coeff = model.coef_.copy()
  residual = L - np.maximum(np.minimum(model.predict(A), 1.), 0.)
  residual = residual.dot(residual)

  import pdb;pdb.set_trace()
  sin_term = np.sqrt((1. - ori_ndotl) * 0.5)
  A = np.empty((ori_ndotl.size, 2 * K))
  ndotlk = ori_ndotl.copy()
  for k in xrange(K):
    A[:,k] = ndotlk
    A[:,k+K] = sin_term * ndotlk
    ndotlk *= ori_ndotl

  import pdb;pdb.set_trace()
  coe=0.5/(1-min(ori_r)/max(ori_r))
  falloffs=coe*(-ori_r/max(ori_r)+1)+1.5
  A *= np.power(ori_r, -falloffs)[:,np.newaxis]

  T=np.maximum(np.minimum(model.predict(A), 1.), 0.)
  
 
  skimage.io.imsave('test3.png',T.reshape(height,width))
  return coeff, residual

# Cook-Torrance BRDF cost function
# variables to solve for:
#   0: scaling of diffuse term (size/intensity of light source)
#   1: scaling of Cook-Torrance term (size/intensity of light source) times
#      Fresnel term [(n - 1) / (n + 1)]^2, where n is the index of refraction
#   2: 1 / m^2; (MSE of slope of facets) from Cook-Torrance
def fit_cook_torrance(L, r, ndotl, falloff=2.):
  ndotlsq = ndotl * ndotl

  G = np.minimum(2 * ndotlsq, 1.) # geometric attenuation term

  # constant factor in the specularity term
  spec_const = G / (np.pi * ndotl**5)
  #neg_tan_arccos_ndotl_sq = np.tan(np.arccos(ndotl))
  #neg_tan_arccos_ndotl_sq *= -neg_tan_arccos_ndotl_sq
  neg_tan_arccos_ndotl_sq = 1. - 1. / ndotlsq

  r_term = np.power(r, -falloff) # cos(theta) / r^m

  def f(X):
    f.est = r_term * (X[0] * ndotl +
        X[1] * spec_const * X[2] * np.exp(neg_tan_arccos_ndotl_sq * X[2]))

    #f.est = np.minimum(f.est, 1.) # cap the estimated intensity

    return np.sum((L - f.est)**2)

  lambertian_alpha = fit_lambertian(L, r, ndotl)[0]
  init_params = np.ones(3)
  init_params[0] = 0.5 * lambertian_alpha
  init_params[1] = 0.5 * lambertian_alpha * 0.04 # taking n = 1.5
  init_params[2] = 100. # (inverse squared) initial microfacet slope

  bounds = [(None, None)] * len(init_params)
  bounds[0] = (0, None)
  bounds[1] = (0, None) # diffuse/specular tradeoff coefficient
  bounds[2] = (1e-6, None) # mean-squared facet slope

  result = minimize(f, init_params, method='L-BFGS-B', bounds=bounds,
                    options={'disp': False, 'iprint': -1})

  return result.x, result.fun

#
#
#

# L: length-N array of image intensities
# r: length-N array of depth values
# ndotl: length-N array of cos(theta) values
# model_func: function that fits and returns parameters to a reflection model;
#             the signature should be model_func(L, r, ndotl, *args)
# args: additional arguments to the reflectance model function
def fit_with_falloff(L, r, ndotl, model_func, *args):
  #
  # perform model estimation
  #

  m_min, m_max = 0., 2. # TODO

  rm_min, residual_min = model_func(L, r, ndotl, *args, falloff=m_min)
  rm_max, residual_max = model_func(L, r, ndotl, *args, falloff=m_max)

  while m_max - m_min > 0.001: # TODO
    m = 0.5 * (m_max + m_min)
    rm, residual = model_func(L, r, ndotl, *args, falloff=m)
    #print m_min, m, m_max
    #print '  ', residual_min, residual, residual_max

    # check linear possibilities
    if residual_min < residual and residual <= residual_max:
      m_est, m_max, rm_max, residual_max = m, m, rm, residual
    elif residual_min >= residual and residual > residual_max:
      m_est, m_min, rm_min, residual_min = m, m, rm, residual
    else:
      # fit a quadratic to our 3 residuals and find the predicted minimum
      A = np.ones((3, 3))
      A[:,1] = (m_min, m, m_max)
      A[:,0] = A[:,1] * A[:,1]
      a, b, c = np.linalg.inv(A).dot((residual_min, residual, residual_max))
      m_est = -0.5 * b / a # estimated minimum

      if m_est == m:
        #print '==', m_est
        break

      rm_est, residual_est = model_func(L, r, ndotl, *args, falloff=m_est)
      #print '::', m_est, rm_est, residual, residual_est

      if residual_est < residual: # was the minimum actually lower?
        if m_est < m:
          m_max, rm_max, residual_max = m, rm, residual
        elif m_est > m:
          m_min, rm_min, residual_min = m, rm, residual
      else: # otherwise, use the point as a new bound
        if m_est > m:
          m_max, rm_max, residual_max = m_est, rm_est, residual_est
        elif m_est < m:
          m_min, rm_min, residual_min = m_est, rm_est, residual_est

  # just in case, fit one more time
  if residual_min < residual:
    m_est, residual = m_min, residual_min
  if residual_max < residual:
    m_est, residual = m_max, residual_max
  reflectance_model, residual = model_func(L, r, ndotl, *args, falloff=m_est)

  return m_est, reflectance_model, residual

#
#
#

#
def apply_lambertian(r, ndotl, alpha, falloff=2.):
  L = alpha * ndotl * np.power(r, -falloff)
  return L

#
def apply_oren_nayar(r, ndotl, model, falloff=2.):
  L = model[0] * ndotl + model[1] * (1. - ndotl * ndotl)
  #L = model[0] * np.power(r, -falloff) * (
  #    (1. - 0.5 * (model[1] / (model[1] + 0.57))) * ndotl +
  #    0.45 * model[1] / (model[1] + 0.09) * (1. - ndotl * ndotl))
  return L

#
def apply_phong(r, ndotl, model, falloff=2.):
  L = (model[0] * ndotl + model[1] * np.power(ndotl, model[2]))
  L *= np.power(r, -falloff)
  return L

#
def apply_power_model(r, ndotl, model, falloff=2.):
  num_coeff = len(model) / 2
  sin_term = np.sqrt((1. - ndotl) * 0.5)

  L = np.zeros_like(ndotl)
  ndotlk = ndotl.copy()
  for ak, bk in izip(model[:num_coeff], model[num_coeff:]):
    L += (ak + bk * sin_term) * ndotlk
    ndotlk *= ndotl

  L *= np.power(r, -falloff)

  return L

#
def apply_cook_torrance(r, ndotl, model, falloff=2.):
  ndotlsq = ndotl * ndotl

  G = np.minimum(2 * ndotlsq, 1.) # geometric attenuation term

  # constant factor in the specularity term
  spec_const = G / (np.pi * ndotl**5)
  neg_tan_arccos_ndotl_sq = 1. - 1. / ndotlsq

  L = (model[0] * ndotl + model[1] *
      spec_const * model[2] * np.exp(neg_tan_arccos_ndotl_sq * model[2]))

  L *= np.power(r, -falloff)

  return L

# model_name should be one of:
#   ["LAMBERTIAN_MODEL", "OREN_NAYAR_MODEL", "PHONG_MODEL", "POWER_MODEL",
#    "COOK_TORRANCE_MODEL"]
def save_reflectance_model(output_file, model_name, residual, model_params,
                           falloff, *extra_params):
  with open(output_file, 'w') as f:
    print>>f, '# <MODEL_NAME> <residual> <extra params> <falloff>',
    print>>f, '<model parameters>'
    print>>f, model_name
    print>>f, residual
    print>>f, ' '.join(map(str, extra_params))
    print>>f, falloff
    print>>f, ' '.join('%.8f' % x for x in model_params)

# outputs:
#   model_type (int): enum value of the given model
#   extra_params (list of strings): if any exist; otherwise, an empty list
#   reflectance model parameters, falloff
def load_reflectance_model(model_file):
  with open(model_file, 'r') as f:
    f.readline() # header
    model_type = getattr(sfs, f.readline().strip())
    f.readline() # residual
    extra_params = f.readline().split()
    falloff = float(f.readline())
    model_params = np.fromstring(f.readline(), sep=' ')
    
  return model_type, extra_params, model_params, falloff

