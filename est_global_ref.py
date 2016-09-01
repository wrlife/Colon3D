import numpy as np
import os
import skimage, skimage.io, skimage.color
import reflectance_models
import sfs
import util

from itertools import izip
from scene_manager import SceneManager
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from skimage.draw import circle

# general algorithm paramters
MAX_NUM_ITER = 10
CONVERGENCE_THRESHOLD = 1e-1 # average change in z over entire image

# warping parameters
KD_TREE_KNN = 10 # number of nearest neighbors to use

# reflectance model fit parameters
MAX_POINT_DISTANCE = 7 # we only fit our model to pixels who are within this
                       # maximum distance, in pixels, from any 2D feature point

# SFS parameters
MAX_NUM_SFS_ITER = 2000
INIT_Z = 500.
SFS_CONVERGENCE_THRESHOLD = 1e-3 # average change in z over entire image
VDERIV_SIGMA = 0.1 # weighting sigma for taking image derivates

# display parameters and figure numbers
WAIT_TIME = 0.001 # time to pause between display updates (needs to be non-zero)


#
#
#

# bilinearly interpolates depth values for given (non-integer) 2D positions
# on a surface
def get_estimated_r(S, points2D_image):
  r_est = np.empty(points2D_image.shape[0])

  for k, (u, v) in enumerate(points2D_image):
    # bilinear interpolation of distances for the fixed 2D points on the current
    # estimated surface
    j0, i0 = int(u), int(v) # upper left pixel for the (u,v) coordinate
    udiff, vdiff = u - j0, v - i0 # distance of sub-pixel coord. to upper left
    p = (udiff * vdiff * S[i0,j0,:] + # this is just the bilinear-weighted sum
      udiff * (1 - vdiff) * S[i0+1,j0,:] +
      (1 - udiff) * vdiff * S[i0,j0,:] +
      (1 - udiff) * (1 - vdiff) * S[i0+1,j0+1,:])

    r_est[k] = np.linalg.norm(p)

  return r_est

def nearest_neighbor_warp(weights, idxs, points2D_image, r_fixed, S):
  # calculate corrective ratios as a weighted sum, where the corrective ratios
  # relate the fixed to estimated depths
  r_ratios = r_fixed / get_estimated_r(S, points2D_image)

  w = np.sum(weights * r_ratios[idxs], axis=-1) / np.sum(weights, axis=-1)
  w = gaussian_filter(w, 7)

  # calculate corrective ratios as a weighted sum
  S *= w[:,:,np.newaxis]

  return S, r_ratios

#
#
#

def fit_reflectance_model(model_type, L, r, ndotl, fit_falloff, *extra_params):
  # determine which reflectance model function to use
  if model_type == sfs.LAMBERTIAN_MODEL:
    model_func = reflectance_models.fit_lambertian
  elif model_type == sfs.OREN_NAYAR_MODEL:
    model_func = reflectance_models.fit_oren_nayar
  elif model_type == sfs.PHONG_MODEL:
    model_func = reflectance_models.fit_phong
  elif model_type == sfs.COOK_TORRANCE_MODEL:
    model_func = reflectance_models.fit_cook_torrance
  elif model_type == sfs.POWER_MODEL:
    model_func = reflectance_models.fit_power_model

  if not fit_falloff:
    falloff = 2.
    model_params, residual = model_func(L, r, ndotl, *extra_params)
  else:
    falloff, model_params, residual = reflectance_models.fit_with_falloff(
        L, r, ndotl, model_func, *extra_params)

  print 'Residual =', residual

  return falloff, model_params, residual


def estimate_overall_ref_model(colmap_folder,min_track_len,min_tri_angle,max_tri_angle,image_path):

	print 'Loading COLMAP data'
	scene_manager = SceneManager(colmap_folder)
	scene_manager.load_cameras()
	scene_manager.load_images()

	images=['frame0859.jpg', 'frame0867.jpg', 'frame0875.jpg', 'frame0883.jpg', 'frame0891.jpg']

	_L=np.array([])
	_r=np.array([])
	_ndotl=np.array([])

	for image_name in images: 	

	  	  
		  image_id = scene_manager.get_image_id_from_name(image_name)
		  image = scene_manager.images[image_id]
		  camera = scene_manager.get_camera(image.camera_id)

		  # image pose
		  R = util.quaternion_to_rotation_matrix(image.qvec)

		  print 'Loading 3D points'
		  scene_manager.load_points3D()
		  scene_manager.filter_points3D(min_track_len,
		    min_tri_angle=min_tri_angle, max_tri_angle=max_tri_angle,
		    image_list=set([image_id]))

		  points3D, points2D = scene_manager.get_points3D(image_id)
		  points3D = points3D.dot(R.T) + image.tvec[np.newaxis,:]

		  # need to remove redundant points
		  # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
		  points2D_view = np.ascontiguousarray(points2D).view(
		    np.dtype((np.void, points2D.dtype.itemsize * points2D.shape[1])))
		  _, idx = np.unique(points2D_view, return_index=True)

		  points2D, points3D = points2D[idx], points3D[idx]

		  # further rule out any points too close to the image border (for bilinear
		  # interpolation)
		  mask = (
		      (points2D[:,0] < camera.width - 1) & (points2D[:,1] < camera.height - 1))
		  points2D, points3D = points2D[mask], points3D[mask]

		  points2D_image = points2D.copy() # coordinates in image space
		  points2D = np.hstack((points2D, np.ones((points2D.shape[0], 1))))
		  points2D = points2D.dot(np.linalg.inv(camera.get_camera_matrix()).T)[:,:2]

		  print len(points3D), 'total points'

		  # load image
		  #image_file = scene_manager.image_path + image.name
		  image_file = image_path + image.name
		  im_rgb = skimage.img_as_float(skimage.io.imread(image_file)) # color image
		  L = skimage.color.rgb2lab(im_rgb)[:,:,0] * 0.01
		  L = np.maximum(L, 1e-6) # unfortunately, can't have black pixels, since we
		                          # divide by L
		  # initial values on unit sphere
		  x, y = camera.get_image_grid()

		  print 'Computing nearest neighbors'
		  kdtree = KDTree(points2D)
		  weights, nn_idxs = kdtree.query(np.c_[x.ravel(),y.ravel()], KD_TREE_KNN)
		  weights = weights.reshape(camera.height, camera.width, KD_TREE_KNN)
		  nn_idxs = nn_idxs.reshape(camera.height, camera.width, KD_TREE_KNN)

		  # turn distances into weights for the nearest neighbors
		  np.exp(-weights, weights) # in-place

		  # create initial surface on unit sphere
		  S0 = np.dstack((x, y, np.ones_like(x)))
		  S0 /= np.linalg.norm(S0, axis=-1)[:,:,np.newaxis]

		  r_fixed = np.linalg.norm(points3D, axis=-1) # fixed 3D depths

		  S = S0.copy()
		  z = S0[:,:,2]
		  S, r_ratios = nearest_neighbor_warp(weights, nn_idxs,
            points2D_image, r_fixed, util.generate_surface(camera, z))
		  z_est = np.maximum(S[:,:,2], 1e-6)
		  S = util.generate_surface(camera, z_est)
		  r = np.linalg.norm(S, axis=-1)
		  ndotl = util.calculate_ndotl(camera, S)
		  if _L.size == 0:
		  	_L=L.ravel()
		  	_r=r.ravel()
		  	_ndotl=ndotl.ravel()
		  else:
		  	_L=np.append(_L,L.ravel())
		  	_r=np.append(_r,r.ravel())
		  	_ndotl=np.append(_ndotl,ndotl.ravel())


	#
	falloff, model_params, residual = fit_reflectance_model(sfs.POWER_MODEL,
      _L, _r, _ndotl, False, 5)		  
	import pdb;pdb.set_trace();