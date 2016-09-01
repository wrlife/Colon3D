#import cv2
import numpy as np
import util

from itertools import combinations, izip

class Camera:
  #
  #
  #
  def __init__(self, type_, width_, height_, params):
    self.width = width_
    self.height = height_

    if type_ == 'SIMPLE_PINHOLE':
      self.fx, self.cx, self.cy = params
      self.fy = self.fx
      self.has_distortion = False
    elif type_ == 'PINHOLE':
      self.fx, self.fy, self.cx, self.cy = params
      self.has_distortion = False
    elif type_ == 'SIMPLE_RADIAL':
      self.fx, self.cx, self.cy, self.k1 = params
      self.fy, self.k2, self.p1, self.p2 = self.fx, 0, 0, 0
      self.has_distortion = True
    elif type_ == 'OPENCV':
      self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2 = \
        params
      self.has_distortion = True
    else:
      # TODO: not supporting other camera types, currently
      raise Exception('Camera type not supported')


  #
  # return an (x, y) image grid for this camera
  #
  def get_camera_matrix(self):
    return np.array(((self.fx, 0, self.cx), (0, self.fy, self.cy), (0, 0, 1)))


  #
  # return an (x, y) image grid for this camera
  #
  def get_image_grid(self):
    return np.meshgrid(
      (np.arange(self.width)  - self.cx) / self.fx,
      (np.arange(self.height) - self.cy) / self.fy)


  #
  # x: array of shape (N,2) or (2,)
  #
  def undistort_points(self, x):
    if not self.has_distortion:
      return x

    # normalize points
    x = np.atleast_2d(x)
    x -= np.array([[self.cx, self.cy]])
    x /= np.array([[self.fx, self.fy]])
      
    p = np.array([self.p1, self.p2])
    xx = x.copy()
    
    for _ in xrange(20):
      xx2 = xx * xx 
      xy = (xx[:,0] * xx[:,1])[:,np.newaxis]
      r2 = (xx2[:,0] + xx2[:,1])[:,np.newaxis]
      radial = r2 * (self.k1 + self.k2 * r2)
    
      xx = x - (xx * radial + 2 * xy * p.T + (r2 + 2 * xx2) * p[::-1].T)
      
    # de-normalize
    xx *= np.array([[self.fx, self.fy]])
    xx += np.array([[self.cx, self.cy]])

    return xx

#  def undistort_image(self, image):
#    if not self.has_distortion:
#      return image
#
#    return cv2.undistort(image, self.get_camera_matrix(),
#                         np.array([self.k1, self.k2, self.p1, self.p2]))

class Image:
  #
  #
  #
  def __init__(self, name_, camera_id_, qvec_, tvec_):
    self.name = name_
    self.camera_id = camera_id_
    self.qvec = qvec_
    self.tvec = tvec_
    self.points2D = np.array([])
    self.point3D_ids = np.array([], dtype=np.int)


class SceneManager:
  #
  #
  #
  def __init__(self, colmap_results_folder):
    self.folder = colmap_results_folder
    #if not self.folder.endswith('/'):
    #  self.folder += '/'

    self.image_path = None
    self.load_colmap_project_file()

    self.cameras = dict()
    self.images = dict()

    # Nx3 array of point3D xyz's
    self.points3D = np.zeros((0, 3))

    # point3D_id => index in self.points3D
    self.point3D_id_to_point3D_idx = dict()

    # point3D_id => set(image_id
    self.point3D_id_to_image_id = dict()

    self.point3D_colors = np.zeros((0, 3))
    self.point3D_errors = np.zeros(0)


  #
  #
  #
  def load_colmap_project_file(self, project_file=None):
    if project_file is None:
      project_file = self.folder + 'project.ini'

    self.image_path = None

    with open(project_file, 'r') as f:
      for line in iter(f.readline, ''):
        # TODO: supporting old colmap format (with '-'); this is deprecated
        if line.startswith('image_path') or line.startswith('image-path'):
          self.image_path = line[11:].strip()
          break

    assert(self.image_path is not None)

    if not self.image_path.endswith('/'):
      self.image_path += '/'
      

  #
  #
  #
  def load_cameras(self, input_file=None):
    if input_file is None:
      input_file = self.folder + 'cameras.txt'

    self.cameras = dict()

    with open(input_file, 'r') as f:
      for line in iter(lambda: f.readline().strip(), ''):
        if not line or line.startswith('#'):
          continue

        data = line.split()
        self.cameras[int(data[0])] = Camera(
          data[1], int(data[2]), int(data[3]), map(float, data[4:]))


  #
  #
  #
  def load_images(self, input_file=None):
    if input_file is None:
      input_file = self.folder + 'images.txt'
    
    self.images = dict()

    with open(input_file, 'r') as f:
      is_camera_description_line = False

      for line in iter(lambda: f.readline().strip(), ''):
        if not line or line.startswith('#'):
          continue

        is_camera_description_line = not is_camera_description_line

        data = line.split()

        if is_camera_description_line:
          image_id = int(data[0])
          image = Image(data[-1], int(data[-2]),
                        np.array(map(float, data[1:5])),
                        np.array(map(float, data[5:8])))
        else:
          image.points2D = np.array(
            [map(float, data[::3]), map(float, data[1::3])]).T
          image.point3D_ids = np.array(map(int, data[2::3]))

          mask = (image.point3D_ids != -1)
          image.points2D = image.points2D[mask]
          image.point3D_ids = image.point3D_ids[mask]
          self.images[image_id] = image


  #
  #
  #
  def load_points3D(self, input_file=None):
    if input_file is None:
      input_file = self.folder + 'points3D.txt'

    self.points3D = []
    self.point3D_colors = []
    self.point3D_id_to_point3D_idx = dict()
    self.point3D_id_to_image_id = dict()
    self.point3D_errors = []

    with open(input_file, 'r') as f:
      for line in iter(lambda: f.readline().strip(), ''):
        if not line or line.startswith('#'):
          continue

        data = line.split()
        point3D_id = int(data[0])

        self.point3D_id_to_point3D_idx[point3D_id] = len(self.points3D)
        self.points3D.append(map(float, data[1:4]))
        self.point3D_colors.append(map(float, data[4:7]))
        self.point3D_errors.append(float(data[7]))

        self.point3D_id_to_image_id[point3D_id] = set(
          int(image_id) for image_id in data[8::2])

    self.points3D = np.array(self.points3D)
    self.point3D_colors = np.array(self.point3D_colors)
    self.point3D_errors = np.array(self.point3D_errors)


  #
  # return the image id associated with a given image file
  #
  def get_image_id_from_name(self, image_name):
    for image_id, image in self.images.iteritems():
      if image.name == image_name:
        return image_id


  #
  #
  #
  def get_camera(self, camera_id):
    return self.cameras[camera_id]


  #
  #
  #
  def get_points3D(self, image_id, return_points2D=True, return_colors=False):
    image = self.images[image_id]
    point3D_idxs = np.array([self.point3D_id_to_point3D_idx[point3D_id]
      for point3D_id in image.point3D_ids])
    mask = (point3D_idxs != -1)
    point3D_idxs = point3D_idxs[mask]
    result = [self.points3D[point3D_idxs,:]]

    if return_points2D:
      result += [image.points2D[mask]]
    if return_colors:
      result += [self.point3D_colors[point3D_idxs,:]]

    return result if len(result) > 1 else result[0]

  #
  # project *all* 3D points into image, return their projection coordinates,
  # as well as their 3D positions
  #
  def get_viewed_points(self, image_id):
    image = self.images[image_id]

    # get unfiltered points
    point3D_idxs = set(self.point3D_id_to_point3D_idx.itervalues())
    point3D_idxs.discard(-1)
    point3D_idxs = list(point3D_idxs)
    points3D = self.points3D[point3D_idxs,:]

    # orient points relative to camera
    R = util.quaternion_to_rotation_matrix(image.qvec)
    points3D = points3D.dot(R.T) + image.tvec[np.newaxis,:]
    points3D = points3D[points3D[:,2] > 0,:] # keep points in front of camera

    # put points into image coordinates
    camera = self.cameras[image.camera_id]
    points2D = points3D.dot(camera.get_camera_matrix().T)
    points2D = points2D[:,:2] / points2D[:,2][:,np.newaxis]

    # keep points that are within the image
    mask = ((points2D[:,0] >= 0) & (points2D[:,1] >= 0) &
      (points2D[:,0] < camera.width - 1) & (points2D[:,1] < camera.height - 1))

    return points2D[mask,:], points3D[mask,:]

  #
  # camera_list: set of cameras whose points we'd like to keep
  #
  def filter_points3D(self, min_track_len=0, max_error=np.inf, min_tri_angle=0,
                      max_tri_angle=180, image_list=set()):
    image_list = set(image_list)

    max_tri_prod = np.cos(np.radians(min_tri_angle))
    min_tri_prod = np.cos(np.radians(max_tri_angle))

    for point3D_id, point3D_idx in self.point3D_id_to_point3D_idx.iteritems():
      image_ids = self.point3D_id_to_image_id[point3D_id]
      
      # check if error and min track length are sufficient, or if none of the
      # selected cameras see the point
      if (len(image_ids) < min_track_len or
            self.point3D_errors[point3D_idx] > max_error or
            image_list and image_list.isdisjoint(image_ids)):
        self.point3D_id_to_point3D_idx[point3D_id] = -1

      # find dot product between all camera viewing rays
      elif min_tri_angle > 0 or max_tri_angle < 180:
        xyz = self.points3D[point3D_idx,:]
        tvecs = np.array(
          [(self.images[image_id].tvec - xyz) for image_id in image_ids])
        tvecs /= np.linalg.norm(tvecs, axis=-1)[:,np.newaxis]

        cos_theta = np.array([u.dot(v) for u,v in combinations(tvecs, 2)])

        # min_prod = cos(maximum viewing angle), and vice versa
        # if maximum viewing angle is too small or too large,
        # don't add this point
        if np.min(cos_theta) > max_tri_prod or np.max(cos_theta) < min_tri_prod:
          self.point3D_id_to_point3D_idx[point3D_id] = -1

