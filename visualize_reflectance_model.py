# Author: True Price <jtprice at cs.unc.edu>

import matplotlib.pyplot as plt
import numpy as np
import reflectance_models
import sfs

def run(model_file):
  model_type, _, model_params, falloff = \
      reflectance_models.load_reflectance_model(model_file)

  grid = np.linspace(1e-6, 50, 1000)
  x, y = np.meshgrid(grid, grid)
  r = np.sqrt(x * x + y * y)
  ndotl = np.cos(np.arctan(y / x))

  if model_type == sfs.LAMBERTIAN_MODEL:
    model_func = reflectance_models.apply_lambertian
  elif model_type == sfs.OREN_NAYAR_MODEL:
    model_func = reflectance_models.apply_oren_nayar
  elif model_type == sfs.PHONG_MODEL:
    model_func = reflectance_models.apply_phong
  elif model_type == sfs.COOK_TORRANCE_MODEL:
    model_func = reflectance_models.apply_cook_torrance
  elif model_type == sfs.POWER_MODEL:
    model_func = reflectance_models.apply_power_model

  L = model_func(r.ravel(), ndotl.ravel(), model_params, falloff)
  L = L.reshape(r.shape)

  plt.pcolormesh(x, y, L, cmap='gray', vmin=0., vmax=1.)
  plt.colorbar()
  plt.axis([grid[0], grid[-1], grid[0], grid[-1]])
  plt.show()

if __name__ == '__main__':
  import sys

  if len(sys.argv) != 2:
    print 'Usage: visualize_reflectance_model.py <model file>'
    exit()

  run(*sys.argv[1:])

