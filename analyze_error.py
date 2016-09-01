# Author: True Price <jtprice at cs.unc.edu>

import numpy as np
import os
import skimage, skimage.io, skimage.color
import util

from itertools import izip
from scene_manager import SceneManager

# colmap-to-world scaling for phantom3 sequence
WORLD_SCALE = 2.60721115767

# maximum error to display, in millimeters
MAX_ERR = 15.

THRESHOLDS = np.linspace(0.5, 5, 10)
HEADER1 = '%s,' + 'mm,,'.join(str(x) for x in THRESHOLDS) + 'mm,'
HEADER2 = ',' + ','.join('mean,std' for _ in THRESHOLDS)

#MODELS = ['lambertian', 'power_model_1', 'power_model_2', 'power_model_3',
    #'power_model_4', 'power_model_5']
#MODELS = ['lambertian', 'power_model_2']
MODELS = ['power_model_5']

#
#
#

def run(colmap_folder, sfs_results_path, gt_path, out_folder):

  #import pdb;pdb.set_trace();
  if not sfs_results_path.endswith('/'):
    sfs_results_path += '/'
  if not gt_path.endswith('/'):
    gt_path += '/'
  if not out_folder.endswith('/'):
    out_folder += '/'

  scene_manager = SceneManager(colmap_folder)
  scene_manager.load_cameras()
  camera = scene_manager.cameras[1] # assume single camera

  # initial values on unit sphere
  x, y = camera.get_image_grid()
  r_scale = np.sqrt(x * x + y * y + 1.)

  image = np.empty((camera.height, camera.width, 3))
  #image[:,:,0] = 1.

  if not os.path.exists(out_folder):
    os.mkdir(out_folder)

  # iterate through rendered BRDFs
  #for brdf_name in os.listdir(sfs_results_path):
  for brdf_name in ['phantom']:
    brdf_folder = sfs_results_path + brdf_name + '/'
    if not os.path.isdir(brdf_folder): continue

    with open(out_folder + brdf_name + '.csv', 'w') as f:
      print>>f, HEADER1 % brdf_name
      print>>f, HEADER2

      # iterate through reflectance models
      #for rm_name in os.listdir(brdf_folder):
      #for model_name in MODELS:
      model_folder = brdf_folder + '/'#model_name + '/'

      # percent of pixels within 1, 2, 5mm of the GT surface
      data = [list() for _ in THRESHOLDS]
###
      datasum = []
###

      # iterate through SFS surfaces
      for filename in os.listdir(model_folder):
        if not filename.endswith('_z.bin'): continue
        name = filename[:-6]


        #import pdb;pdb.set_trace();

        util.save_sfs_ply('%s_%s.ply' % (name, 'est'),
            np.dstack((x, y, np.ones_like(x))) *
            np.fromfile(model_folder + filename, dtype=np.float32).reshape(
                camera.height, camera.width, 1))
#
        util.save_sfs_ply('%s_%s.ply' % (name, 'gt'),
            np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE *
            np.fromfile(gt_path + name+'.bin', dtype=np.float32).reshape(
                camera.height, camera.width, 1))
        #break

        r_est = WORLD_SCALE * r_scale * np.fromfile(
            model_folder + filename, dtype=np.float32).reshape(
                camera.height, camera.width)
        r_gt = r_scale * np.fromfile(
          gt_path + name + '.bin', dtype=np.float32).reshape(
            camera.height, camera.width)

        rdiff = np.abs(r_est - r_gt)
        #rdiff = np.concatenate((
        #  rdiff[:,:20].ravel(), rdiff[:,-20:].ravel(),
        #  rdiff[20:-20,:20].ravel(), rdiff[20:-20,-20:].ravel()))
        #rdiff = rdiff[(r_gt > 5.0) & (r_gt < 20.0)]
        inv_size = 1. / float(rdiff.size)

        for i, t in enumerate(THRESHOLDS):
          data[i].append(np.count_nonzero(rdiff < t) * inv_size)
          #data[i].append(np.mean((rdiff / r_gt)[rdiff < t]))
          #data[i].append(np.mean((r_est / r_gt)))
###
        datasum.append(1. - (r_est / r_gt).ravel())
      datasum = np.concatenate(datasum)
      print np.mean(datasum), np.std(datasum)
###

      data = np.array(data)
      data = np.mean(data, axis=1), np.std(data, axis=1)
      model_name='power'
      print>>f, model_name + ',' + ','.join('%f,%f' % d for d in izip(*data))

#
#
#

if __name__ == '__main__':
  import sys

  if len(sys.argv) != 5:
    print 'Usage: analyze_error.py <colmap project folder> <SFS results path>'
    print '       <ground truth path> <output folder>'
    exit()

  run(*sys.argv[1:])

