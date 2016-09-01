# Author: True Price <jtprice at cs.unc.edu>

#import matplotlib.pyplot as plt
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



import Tkinter as Tk
import vtk
from itertools import izip
from VTKViewer.VTKViewer import VTKViewer
from VTKViewer.VTKViewerTk import VTKViewerTk # debug
from rotation import Quaternion
from numpy.linalg import inv


import est_global_ref

from sfs_cuda import compute_nn
#
#
#

# general algorithm paramters
MAX_NUM_ITER = 3
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

# linear spacing on theta
NUM_H_LUT_BINS = 1000


WORLD_SCALE = 2.60721115767


#
#
#


def compute_H_LUT(model_type, model_params, num_h_lut_bins):
  _, model_func = model_func_from_type(model_type)

  ndotl = np.linspace(0., 1., num_h_lut_bins + 1.)[:-1]
  inv_ndotl_step = 1. / ndotl[1]

  H_lut = model_func(1, ndotl, model_params) # using a dummy r value

  # each value in dH_lut is the maximum value of (dH/dp) for p in [0, ndotl]
  dH_lut = np.empty_like(H_lut)
  dH_lut[0] = H_lut[1] * inv_ndotl_step
  dH_lut[1:-1] = (H_lut[2:] - H_lut[:-2]) * (inv_ndotl_step * 0.5)
  dH_lut[-1] = (
      model_func(1, np.array([1.]), model_params) - H_lut[-1]) * inv_ndotl_step
  dH_lut = np.maximum.accumulate(dH_lut)

  return H_lut, dH_lut


def model_func_from_type(model_type):
  if model_type == sfs.LAMBERTIAN_MODEL:
    fit_func = reflectance_models.fit_lambertian
    apply_func = reflectance_models.apply_lambertian
  elif model_type == sfs.OREN_NAYAR_MODEL:
    fit_func = reflectance_models.fit_oren_nayar
    apply_func = reflectance_models.apply_oren_nayar
  elif model_type == sfs.PHONG_MODEL:
    fit_func = reflectance_models.fit_phong
    apply_func = reflectance_models.apply_phong
  elif model_type == sfs.COOK_TORRANCE_MODEL:
    fit_func = reflectance_models.fit_cook_torrance
    apply_func = reflectance_models.apply_cook_torrance
  elif model_type == sfs.POWER_MODEL:
    fit_func = reflectance_models.fit_power_model
    apply_func = reflectance_models.apply_power_model

  return fit_func, apply_func

#
#
#
#Rui-04/07/2016##
#Obtain Ref surface from fused surface
def extract_depth_map(camera,ref_surf_name,R,image):

  viewer = VTKViewer(width=camera.width, height=camera.height)
  viewer.toggle_crosshair()
  viewer.set_camera_params(camera.width, camera.height,
                           camera.fx, camera.fy, camera.cx, camera.cy)
  viewer.load_ply(ref_surf_name)  
  viewer.set_pose(inv(R), -inv(R).dot(image.tvec)) 
  ndepth=viewer.get_z_values() 
  S = util.generate_surface(camera, ndepth)
  z_est = np.maximum(S[:,:,2], 1e-6)
  z_est[(z_est>INIT_Z)]=1000
  return z_est

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

def fit_reflectance_model(model_type, L, ori_r, r, ori_ndotl,ndotl, fit_falloff,width,height, *extra_params):
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
    falloff =1.5
    model_params, residual = model_func(L,ori_r, r,ori_ndotl,ndotl,width,height, *extra_params)
  else:
    #import pdb;pdb.set_trace()
    falloff, model_params, residual = reflectance_models.fit_with_falloff(L,ori_r, r,ori_ndotl,ndotl,width,height, model_func, *extra_params)

  print 'Residual =', residual
  print 'Falloff = ', falloff

  return falloff, model_params, residual

#
#
#

def run_sfs(H_lut,dH_lut,camera, L, lambdas, z_est, model_type, model_params, falloff, vmask,
            use_image_weighted_derivatives=True, display=True):
  print 'Running SFS'

  # some initialization
  x, y = camera.get_image_grid()
  l = np.dstack((x, y, np.ones_like(x))) # lighting direction vector
  inv_sqrt_xsq_ysq_1 = 1. / np.linalg.norm(l, axis=-1)
  l *= inv_sqrt_xsq_ysq_1[:,:,np.newaxis]

  # normalize the luminance values according to the lighting assumptions
  Lhat = np.power(inv_sqrt_xsq_ysq_1, falloff) / L

  v = np.empty((camera.height, camera.width))

  # compute image-intensity-based derivative weights
  inv_vderiv_sq = 1. / (VDERIV_SIGMA * VDERIV_SIGMA)

  if use_image_weighted_derivatives:
    wxfw, wxbk = np.zeros_like(L), np.zeros_like(L)
    wxfw[:,:-1] = np.exp(-inv_vderiv_sq * (L[:,1:] - L[:,:-1])**2)
    wxbk[:,1:] = wxfw[:,:-1]
    wxtotal = wxfw + wxbk
    wxfw /= wxtotal
    wxbk /= wxtotal

    wyfw, wybk = np.zeros_like(L), np.zeros_like(L)
    wyfw[:-1,:] = np.exp(-inv_vderiv_sq * (L[1:,:] - L[:-1,:])**2)
    wybk[1:,:] = wyfw[:-1,:]
    wytotal = wyfw + wybk
    wyfw /= wytotal
    wybk /= wytotal
  else:
    wxfw, wxbk = 0.5 * np.ones_like(L), 0.5 * np.ones_like(L)
    wyfw, wybk = 0.5 * np.ones_like(L), 0.5 * np.ones_like(L)
  #import pdb;pdb.set_trace()
  # initialize module
  z_est = np.power(z_est, falloff)
  sfs.initialize(H_lut,dH_lut,L, Lhat, v, x, y, z_est, inv_sqrt_xsq_ysq_1,wxfw, wxbk, wyfw, wybk, lambdas, camera.fx, camera.fy,vmask)
                 
  sfs.set_model(model_type, model_params, falloff)

  # run the algorithm

  num_iter = 0

  z_old = np.ones_like(v) * INIT_Z
  v[:] = np.log(INIT_Z)

  while num_iter < MAX_NUM_SFS_ITER:
    num_iter += 1

    sfs.step()

    # boundary conditions:
    #   a: p+ - p- = 0
    #   b: p+ + p- = 0
    wfw, wbk = wxfw[:,1], wxbk[:,1]
    v0_a = (v[:,1] - wfw * v[:,2]) / wbk
    v0_b = (v[:,2] - v[:,1]) * wfw / wbk + v[:,1]
    v[:,0] = np.minimum(np.minimum(v0_a, v0_b), v[:,0])

    wfw, wbk = wxfw[:,-2], wxbk[:,-2]
    v0_a = (v[:,-2] - wbk * v[:,-3]) / wfw
    v0_b = (v[:,-3] - v[:,-2]) * wbk / wfw + v[:,-2]
    v[:,-1] = np.minimum(np.minimum(v0_a, v0_b), v[:,-1])

    wfw, wbk = wyfw[1,:], wybk[1,:]
    v0_a = (v[1,:] - wfw * v[2,:]) / wbk
    v0_b = (v[2,:] - v[1,:]) * wfw / wbk + v[1,:]
    v[0,:] = np.minimum(np.minimum(v0_a, v0_b), v[0,:])

    wfw, wbk = wyfw[-2,:], wybk[-2,:]
    v0_a = (v[-2,:] - wbk * v[-3,:]) / wfw
    v0_b = (v[-3,:] - v[-2,:]) * wbk / wfw + v[-2,:]
    v[-1,:] = np.minimum(np.minimum(v0_a, v0_b), v[-1,:])

    z = np.exp(v)

#    if display:
#      # show z values
#      plt.clf()
#      plt.imshow(z)
#      plt.colorbar()
#
#      plt.waitforbuttonpress(WAIT_TIME)

    diff = np.sum(np.abs(z_old - z))
    #print num_iter, diff, ':', np.min(z), np.max(z)
    if diff < SFS_CONVERGENCE_THRESHOLD * camera.height * camera.width:
      break

    z_old = z

  print num_iter, diff, ':', np.min(z), np.max(z)

  # return the new depth values
  return z

#
#
#

def run(image_name, image_path, colmap_folder, out_folder, min_track_len,
        min_tri_angle, max_tri_angle, ref_surf_name, max_num_points=None,
        estimate_falloff=True, use_image_weighted_derivatives=True,get_initial_warp=True):




  min_track_len = int(min_track_len)
  min_tri_angle = int(min_tri_angle)
  max_tri_angle = int(max_tri_angle)

  #est_global_ref.estimate_overall_ref_model(colmap_folder,min_track_len,min_tri_angle,max_tri_angle,image_path)
  
  try:
    max_num_points = int(max_num_points)
  except:
    max_num_points = None

  get_initial_warp = (get_initial_warp not in ('False', 'false'))

  estimate_falloff = (estimate_falloff not in ('False', 'false'))
  use_image_weighted_derivatives = (use_image_weighted_derivatives not in ('False', 'false'))


  #if not image_path.endswith('/'):
  #  image_path += '/'

  print 'Loading COLMAP data'
  scene_manager = SceneManager(colmap_folder)
  scene_manager.load_cameras()
  scene_manager.load_images()
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

  if (max_num_points is not None and max_num_points > 0 and
      max_num_points < len(points2D)):
    np.random.seed(0) # fix the "random" points selected
    selected_points = np.random.choice(len(points2D), max_num_points, False)
    points2D = points2D[selected_points]
    points2D_image = points2D_image[selected_points]
    points3D = points3D[selected_points]

  print len(points3D), 'total points'

  #perturb points

  #import pdb;pdb.set_trace()
  #perturb_points=np.random.choice(len(points2D),max_num_points*0.3,False)
  #randomperturb=np.random.rand(perturb_points.size)*2
  #points3D[perturb_points,2]=points3D[perturb_points,2]*randomperturb


  #points3D[2][2]=points3D[2][2]*0.5
  #points3D[4][2]=points3D[4][2]*1.5
  #points3D[8][2]=points3D[8][2]*0.3
  #points3D[9][2]=points3D[9][2]*2.
  # load image
  #image_file = scene_manager.image_path + image.name
  image_file = image_path + image.name
  im_rgb = skimage.img_as_float(skimage.io.imread(image_file)) # color image
  L = skimage.color.rgb2lab(im_rgb)[:,:,0] * 0.01

  #import pdb;pdb.set_trace()

  #skimage.io.imsave('test.png',L)

  L = np.maximum(L, 1e-6) # unfortunately, can't have black pixels, since we
                          # divide by L

  # initial values on unit sphere
  x, y = camera.get_image_grid()

  print 'Computing nearest neighbors'

  #import pdb;pdb.set_trace()
  nn_idxs, weights = compute_nn(points2D.astype(np.float32), camera.width,
      camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
  nn_idxs=np.swapaxes(nn_idxs,0,2)
  nn_idxs=np.swapaxes(nn_idxs,0,1)

  weights=np.swapaxes(weights,0,2)
  weights=np.swapaxes(weights,0,1)  
  
  #kdtree = KDTree(points2D)
  #weights, nn_idxs = kdtree.query(np.c_[x.ravel(),y.ravel()], KD_TREE_KNN)
  #weights = weights.reshape(camera.height, camera.width, KD_TREE_KNN)
  #nn_idxs = nn_idxs.reshape(camera.height, camera.width, KD_TREE_KNN)

  # figure out pixel neighborhoods for each point
  #neighborhoods = []
  neighborhood_mask = np.zeros((camera.height, camera.width), dtype=np.bool)
  for v, u in points2D_image:
    rr, cc = circle(int(u), int(v), MAX_POINT_DISTANCE,
        (camera.height, camera.width))
    #neighborhoods.append((rr, cc))
    neighborhood_mask[rr,cc] = True

  # turn distances into weights for the nearest neighbors
  np.exp(-weights, weights) # in-place

  # create initial surface on unit sphere
  S0 = np.dstack((x, y, np.ones_like(x)))
  S0 /= np.linalg.norm(S0, axis=-1)[:,:,np.newaxis]

  r_fixed = np.linalg.norm(points3D, axis=-1) # fixed 3D depths

  specular_mask = (L < 1.)
  vmask=np.zeros_like(S0[:,:,2])
  for k, (u, v) in enumerate(points2D_image):
    j0, i0 = int(u), int(v) # upper left pixel for the (u,v) coordinate
    vmask[i0,j0]=points3D[k,2]

  # iterative SFS algorithm
  # model_type: reflectance model type
  # fit_falloff: True to fit the 1/r^m falloff parameter; False to fix it at 2
  # extra_params: extra parameters to the reflectance model fit function
  def run_iterative_sfs(out_file, model_type, fit_falloff, *extra_params):
    #if os.path.exists(out_file + '_z.bin'): # don't re-run existing results
    #  return

    if model_type == sfs.LAMBERTIAN_MODEL:
      model_name = 'LAMBERTIAN_MODEL'
    elif model_type == sfs.OREN_NAYAR_MODEL:
      model_name = 'OREN_NAYAR_MODEL'
    elif model_type == sfs.PHONG_MODEL:
      model_name = 'PHONG_MODEL'
    elif model_type == sfs.COOK_TORRANCE_MODEL:
      model_name = 'COOK_TORRANCE_MODEL'
    elif model_type == sfs.POWER_MODEL:
      model_name = 'POWER_MODEL'

    if model_type == sfs.POWER_MODEL:
      print '%s_%i' % (model_name, extra_params[0])
    else:
      print model_name

    S = S0.copy()
    z = S0[:,:,2]

    for iteration in xrange(MAX_NUM_ITER):
      print 'Iteration', iteration

      z_old = z

      if get_initial_warp:

      	#z_gt=util.load_point_ply('C:\\Users\\user\\Documents\\UNC\\Research\\ColonProject\\code\\Rui\\SFS_CPU\\frame0859.jpg_gt.ply')
        # warp to 3D points
        if iteration>-1:
        	S, r_ratios = nearest_neighbor_warp(weights, nn_idxs,
        	    points2D_image, r_fixed, util.generate_surface(camera, z))
        	z_est = np.maximum(S[:,:,2], 1e-6)
        	S=util.generate_surface(camera, z_est)

        else:
        	z_est=z

        S=util.generate_surface(camera, z_est)
        #util.save_sfs_ply('warp' + '.ply', S, im_rgb)
        #util.save_xyz('test.xyz',points3D);
        #z=z_est
        #break
        #z_est=z

      else:
        #import pdb;pdb.set_trace()
        #z_est = extract_depth_map(camera,ref_surf_name,R,image)
        z_est=np.fromfile(
          'C:\Users\user\Documents\UNC\Research\ColonProject\code\SFS_Program_from_True\endo_evaluation\gt_surfaces\\frame0859.jpg.bin', dtype=np.float32).reshape(
            camera.height, camera.width)/WORLD_SCALE
        z_est=z_est.astype(float)
        #S, r_ratios = nearest_neighbor_warp(weights, nn_idxs,
        #    points2D_image, r_fixed, util.generate_surface(camera, z_est))
        z_est = np.maximum(z_est[:,:], 1e-6)
        #Sworld = (S - image.tvec[np.newaxis,np.newaxis,:]).dot(R)
        S = util.generate_surface(camera, z_est)
        #util.save_sfs_ply('test' + '.ply', S, im_rgb)
        #util.save_sfs_ply(out_file + '_warp_%i.ply' % iteration, Sworld, im_rgb)
        #import pdb;pdb.set_trace()
        # if we need to, make corrections for non-positive depths
      
        #S = util.generate_surface(camera, z_est)

      mask = (z_est < INIT_Z)

      specular_mask=(L<0.8)
      dark_mask=(L>0.1)
      _mask=np.logical_and(specular_mask,mask)
      _mask=np.logical_and(_mask,dark_mask)
      # fit reflectance model
      r = np.linalg.norm(S, axis=-1)
      ndotl = util.calculate_ndotl(camera, S)
      falloff, model_params, residual = fit_reflectance_model(model_type,
          L[_mask],r.ravel(), r[_mask],ndotl.ravel(), ndotl[_mask], fit_falloff,camera.width,camera.height, *extra_params)
      #r = np.linalg.norm(S[specular_mask], axis=-1)

      #import pdb;pdb.set_trace()
      #model_params=np.array([26.15969874,-27.674055,-12.52426,7.579855,21.9768004,24.3911142,-21.7282996,-19.850894,-11.62229,-4.837014])
      #model_params=np.array([-19.4837,-490.4796,812.4527,-426.09107,139.2602,351.8061,-388.1591,875.5013,-302.4748,-414.4384])
      #falloff = 1.2
      #ndotl = util.calculate_ndotl(camera, S)[specular_mask]
      #falloff, model_params, residual = fit_reflectance_model(model_type,
      #    L[specular_mask], r, ndotl, fit_falloff, *extra_params)
      #r = np.linalg.norm(S[neighborhood_mask], axis=-1)
      #ndotl = util.calculate_ndotl(camera, S)[neighborhood_mask]
      #falloff, model_params, residual = fit_reflectance_model(model_type,
      #    L[neighborhood_mask], r, ndotl, fit_falloff, *extra_params)

      # lambda values reflect our confidence in the current surface: 0
      # corresponds to only using SFS at a pixel, 1 corresponds to equally
      # weighting SFS and the current estimate, and larger values increasingly
      # favor using only the current estimate

      rdiff = np.abs(r_fixed - get_estimated_r(S, points2D_image))
      w = np.log10(r_fixed) - np.log10(rdiff) - np.log10(2.)
      lambdas = (np.sum(weights * w[nn_idxs], axis=-1) /
          np.sum(weights, axis=-1))
      lambdas = np.maximum(lambdas, 0.) # just in case
#
      lambdas[~mask] = 0

      #if iteration == 0: # don't use current estimated surface on first pass
      #lambdas = np.zeros_like(z)
      #else:
      #  r_ratios_postwarp = r_fixed / get_estimated_r(S, points2D_image)
      #  ratio_diff = np.abs(r_ratios_prewarp - r_ratios_postwarp)
      #  ratio_diff[ratio_diff == 0] = 1e-10 # arbitrarily high lambda
      #  feature_lambdas = 1. / ratio_diff
      #  lambdas = (np.sum(weights * feature_lambdas[nn_idxs], axis=-1) /
      #    np.sum(weights, axis=-1))

      # run SFS

      H_lut, dH_lut = compute_H_LUT(model_type, model_params, NUM_H_LUT_BINS)
      #import pdb;pdb.set_trace()
      # run SFS
      #H_lut = np.ascontiguousarray(H_lut.astype(np.float32))
      #dH_lut = np.ascontiguousarray(dH_lut.astype(np.float32))


      z = run_sfs(H_lut,dH_lut,camera, L, lambdas, z_est, model_type, model_params, falloff,vmask,
          use_image_weighted_derivatives)

      # check for convergence
      #diff = np.sum(np.abs(z_old[specular_mask] - z[specular_mask]))
      #if diff < CONVERGENCE_THRESHOLD * camera.height * camera.width:
      #  break
      
      # save the surface
      #S = util.generate_surface(camera, z)
      #S = (S - image.tvec[np.newaxis,np.newaxis,:]).dot(R)
      #util.save_sfs_ply(out_file + '_%i.ply' % iteration, S, im_rgb)
    else:
      print 'DID NOT CONVERGE'

    #import pdb;pdb.set_trace()
    S = util.generate_surface(camera, z)
    #S = (S - image.tvec[np.newaxis,np.newaxis,:]).dot(R)
   
    util.save_sfs_ply(out_file + '.ply', S, im_rgb)
   
    z.astype(np.float32).tofile(out_file + '_z.bin')

    # save the surface
    #S = util.generate_surface(camera, z)
    #S = (S - image.tvec[np.newaxis,np.newaxis,:]).dot(R)
    #S, r_ratios = nearest_neighbor_warp(weights, nn_idxs,
    #    points2D_image, r_fixed, util.generate_surface(camera, z))    
    #util.save_sfs_ply(out_file + '_warped.ply', S, im_rgb)
    #z = np.maximum(S[:,:,2], 1e-6)
    #z.astype(np.float32).tofile(out_file + '_warped_z.bin')

    #reflectance_models.save_reflectance_model(out_file + '_reflectance.txt',
    #    model_name, residual, model_params, falloff, *extra_params)

    print


  # now, actually run SFS
  if not out_folder.endswith('/'):
    out_folder += '/'

  if not os.path.exists(out_folder):
    os.makedirs(out_folder)

#  if not os.path.exists(out_folder + 'lambertian/'):
#    os.mkdir(out_folder + 'lambertian/')
#  run_iterative_sfs(out_folder + 'lambertian/' + image.name,
#      sfs.LAMBERTIAN_MODEL, estimate_falloff)
#  if not os.path.exists(out_folder + 'oren_nayar/'):
#    os.mkdir(out_folder + 'oren_nayar/')
#  run_iterative_sfs(out_folder + 'oren_nayar/' + image.name,
#      sfs.OREN_NAYAR_MODEL, estimate_falloff)

#  if not os.path.exists(out_folder + 'phong/'):
#    os.mkdir(out_folder + 'phong/')
#  run_iterative_sfs(out_folder + 'phong/' + image.name, sfs.PHONG_MODEL,
#      estimate_falloff)
#  if not os.path.exists(out_folder + 'cook_torrance/'):
#    os.mkdir(out_folder + 'cook_torrance/')
#  run_iterative_sfs(out_folder + 'cook_torrance/' + image.name,
#      sfs.COOK_TORRANCE_MODEL, estimate_falloff)

  #for K in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
  #for K in [10, 20, 50]:
  #for K in [1, 2, 3, 4, 5]:

  for K in [100]:
    #if not os.path.exists(out_folder + 'power_model_%i/' % K):
    #  os.mkdir(out_folder + 'power_model_%i/' % K)
    run_iterative_sfs(out_folder + image.name,
        sfs.POWER_MODEL, estimate_falloff, K)

#
#
#

if __name__ == '__main__':
  import sys

  if len(sys.argv) < 13 or len(sys.argv) > 14:
    print 'Usage: run_sfs.py <image name> <image path> <colmap project folder>'
    print '       <output folder> <min track len> <min tri angle>',
    print '<max tri angle> <max num points> <estimate falloff>'
    print '       <use image weighted derivatives>'
    print '       [--display]'
    exit()

  run(*sys.argv[1:])

