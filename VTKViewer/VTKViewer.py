# Author: Adam Aji <aji at cs.unc.edu>
#         True Price <jtprice at cs.unc.edu>

import numpy as np
import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# TODO: will probably move this definition to a utility file in the future
def _vtkMatrix4x4_to_numpy(vtkMat):
  M = np.empty((4, 4)) # numpy representation of P
  for i in xrange(4):
    for j in xrange(4):
      M[i, j] = vtkMat.GetElement(i, j)
  return M

class VTKViewer:
  DEFAULT_WIDTH = 800
  DEFAULT_HEIGHT = 800

  MIN_POINT_SIZE = 0.5
  MAX_POINT_SIZE = 50.
  POINT_SIZE_SCROLL_FACTOR = 1.1

  INIT_NEAR_CLIPPING = 0.1
  FAR_CLIPPING = 1000
  MIN_NEAR_CLIPPING = 0.001
  MAX_NEAR_CLIPPING = 10
  NEAR_CLIPPING_SCROLL_FACTOR = 1.1

  #
  # constructors
  #

  def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               render_window=None):
    if render_window is not None:
      self.render_window = render_window
    else:
      self.render_window = vtk.vtkRenderWindow()
      self.render_window.SetOffScreenRendering(True)

    self._camera = vtk.vtkCamera() # perspective camera

    self._lighting_on = True

    self._render_blocking = False

    self.point_size = 1.

    self._crosshair_actor = None
    self._current_mesh_id = -1
    self._mesh_actors = dict()
    self._landmark_actors = []
    
    self._renderer = vtk.vtkRenderer()
    self._renderer.SetActiveCamera(self._camera)

    # compute default camera params (camera parameters are in normalized device
    # coordinates)
    self.width, self.height = width, height

    self.near_clipping, self.far_clipping = self._camera.GetClippingRange()
    #self._camera.SetClippingRange(self.INIT_NEAR_CLIPPING, self.FAR_CLIPPING)
    #self.near_clipping = self.INIT_NEAR_CLIPPING
    P = self._camera.GetProjectionTransformMatrix(
        self.width / self.height, self.near_clipping, self.far_clipping)
    self._fx, self._fy = P.GetElement(0, 0), P.GetElement(1, 1)
    self._cx, self._cy = P.GetElement(0, 2), P.GetElement(1, 2)

    self.render_window.AddRenderer(self._renderer)

    self._initialize_crosshair()

  #
  # general private functions
  #

  def _actor_from_poly_data(self, poly_data):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

  # returns:
  #   mesh_id:  ID for the added mesh
  def _add_mesh_actor(self, actor):
    actor.GetProperty().SetLighting(self._lighting_on)
    self._current_mesh_id += 1
    self._mesh_actors[self._current_mesh_id] = actor
    self._renderer.AddActor(actor)
    self.update()

    return self._current_mesh_id

  def _add_color_to_actor(self, actor, colors):
    colors = np.asarray(colors)

    if colors.size == 3: # only one color given
      colors = colors.ravel()
      actor.GetProperty().SetColor(colors[0], colors[1], colors[2])
    else: # array of colors given
      colors = np.ascontiguousarray(colors.reshape(-1, colors.shape[-1]).astype(
          np.uint8, subok=True, copy=False))
      colors_array = numpy_to_vtk(colors)
      colors_array.SetName("Colors")

      # to avoid the colors array going out of scope
      setattr(colors_array, '_np_color_array', colors)

      poly_data = actor.GetMapper().GetInput()
      poly_data.GetPointData().SetScalars(colors_array)

  # s: minimum/maximum values for the crosshair
  def _generate_crosshair_points(self, x, y, z, s):
    return np.array(
        ((-s, y, z), (s, y, z), (x, -s, z), (x, s, z), (x, y, -s), (x, y, s)))

  def _get_mesh_polydata(self, mesh_id):
    if mesh_id not in self._mesh_actors:
      return None
    
    return self._mesh_actors[mesh_id].GetMapper().GetInput()

  def _get_z_values(self):
    z = vtk.vtkFloatArray()
    self.render_window.GetZbufferData(
        0, 0, self.width - 1, self.height - 1, z)
    z = np.flipud(
        vtk_to_numpy(z).astype(np.double).reshape(self.height, self.width))

    # convert to perspective projection
    # see: http://stackoverflow.com/questions/6652253/
    # although, that explanation isn't quite right
    z = 2. * z - 1. # rescale from [0,1] to [-1,1]
    z = 2. * self.near_clipping * self.far_clipping / (
        self.far_clipping + self.near_clipping -
        z * (self.far_clipping - self.near_clipping))

    return z

  def _initialize_crosshair(self):
    points = self._generate_crosshair_points(0., 0., 0., 1.)
    lines = np.array([[0, 1], [2, 3], [4, 5]])
    colors = np.array(((255, 0, 0), (0, 255, 0), (0, 0, 255)), dtype=np.uint8)

    mesh_id = self.add_lines(points, lines, colors)

    # remove the crosshair from our mesh list, so that no one deletes it
    self._crosshair_actor = self._mesh_actors.pop(mesh_id)

  # converts a ...X3 numpy array of points into a vtkPoints object
  def _convert_points_array(self, points):
    # directly reference the point cloud data from numpy arrays
    points = np.ascontiguousarray(
      points.reshape(-1, 3).astype(np.double, subok=True, copy=False))
    points_array = numpy_to_vtk(points)

    # to avoid having points potentially go out of scope
    setattr(points_array, '_np_points_array', points)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(points_array)
    vtk_points.SetDataTypeToDouble()

    return vtk_points

  # see set_camera_params() for a full description of what we're doing
  def _update_camera_params(self):
    P = vtk.vtkMatrix4x4() # projection matrix with clipping planes
    P.Zero()
    P.SetElement(0, 0, self._fx)
    P.SetElement(1, 1, self._fy)
    P.SetElement(0, 2, self._cx)
    P.SetElement(1, 2, self._cy)
    P.SetElement(2, 2, -self.near_clipping - self.far_clipping)
    P.SetElement(2, 3, -self.near_clipping * self.far_clipping)
    P.SetElement(3, 2, -1.)

    # first, reset the user transformation matrix
    cameraTransform = vtk.vtkPerspectiveTransform()
    self._camera.SetUserTransform(cameraTransform)

    # current projection matrix for the VTK camera
    Minv = self._camera.GetProjectionTransformMatrix(
        self.width / self.height, self.near_clipping, self.far_clipping)
    Minv.Invert()

    # desired user transform matrix U: UM = P
    U = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(P, Minv, U)

    # and finally update the transform
    cameraTransform.SetMatrix(U)
    self._camera.SetUserTransform(cameraTransform)

    self.render_window.SetSize(self.width, self.height)
    self.update()

  #
  # public functions
  #

  def add_landmark(self, x,y,z, color=(1,0,0), radius=0.5):
    VtkSourceSphere = vtk.vtkSphereSource()
    VtkSourceSphere.SetCenter(x, y, z)
    VtkSourceSphere.SetRadius(radius)
    VtkSourceSphere.SetPhiResolution(360)
    VtkSourceSphere.SetThetaResolution(360)

    VtkMapperSphere = vtk.vtkPolyDataMapper()
    VtkMapperSphere.SetInputConnection(VtkSourceSphere.GetOutputPort())

    VtkActorSphere = vtk.vtkActor()
    VtkActorSphere.SetMapper(VtkMapperSphere)
    VtkActorSphere.GetProperty().SetColor(color[0], color[1], color[2])

    VtkRenderer = vtk.vtkRenderer()
    VtkRenderer.SetBackground(1.0, 1.0, 1.0)
    self._renderer.AddActor(VtkActorSphere)
    self._landmark_actors.append(VtkActorSphere)
    self.update()   

  # H: a 3x3, 3x4, or 4x4 transformation matrix
  # mesh_id: if not specified, applies the transformation to all loaded meshes
  def apply_transformation(self, H, mesh_id=None):
    if mesh_id is None:
      mesh_ids = self._mesh_actors.iterkeys()
    else:
      mesh_ids = [mesh_id]

    for mesh_id in mesh_ids:
      points3D_original = self.get_points(mesh_id)
      points3D = points3D_original.dot(H[:3,:3].T)
      if H.shape[1] == 4: # H is at least 3x4
        points3D += H[:3,3][np.newaxis,:]
      if H.shape[0] == 4: # H is 4x4; compute homogenous element w and rescale
        w = points3D_original.dot(H[3,:3][:,np.newaxis]) + H[3,3]
        points3D /= w
      self.update_points(mesh_id, points3D)

  # points: ...x3 array of numpy points
  # lines: length L array of lines of arbitrary length; lines[i] provides a list
  #   of k_i indices in points (using points.reshape(-1, 3))
  # colors: one color per line (Lx3), or separate colors per vertex (...x3)
  #
  # TODO: this is fairly slow; need to find out a way to map the lines array
  #       directly to a vtkCellArray
  def add_lines(self, points, lines, colors=(255, 0, 0)):
    line_collection = vtk.vtkCellArray()
    line_collection.Allocate(len(lines))
    for line in lines:
      line_collection.InsertNextCell(len(line), line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(self._convert_points_array(points))
    poly_data.SetLines(line_collection)

    actor = self._actor_from_poly_data(poly_data)

    colors = np.asarray(colors)
    if colors.size == 3: # only one color given
      colors = colors.ravel()
      actor.GetProperty().SetColor(colors[0], colors[1], colors[2])
    elif colors.shape[0] == len(lines): # array of line colors
      colors = np.ascontiguousarray(
        colors.astype(np.uint8, subok=True, copy=False))
      colors_array = numpy_to_vtk(colors)
      colors_array.SetName("Colors")

      # to avoid the colors array going out of scope
      setattr(colors_array, '_np_color_array', colors)

      poly_data.GetCellData().SetScalars(colors_array)
    else: # array of vertex colors
      self._add_color_to_actor(actor, colors)

    return self._add_mesh_actor(actor)

  def begin_render(self):
    self._render_blocking = True
    
  def delete_mesh(self, mesh_id):
    if mesh_id in self._mesh_actors:
      actor = self._mesh_actors.pop(mesh_id)
      self._renderer.RemoveActor(actor)
      self.update()

  def clone_mesh(self, mesh_id):
    if mesh_id not in self._mesh_actors:
      return None

    old_actor = self._mesh_actors[mesh_id]
    polyData = vtk.vtkPolyData()
    polyData.DeepCopy(old_actor.GetMapper().GetInput())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(self.point_size)

    return self._add_mesh_actor(actor)

  def finalize_render(self):
    self._render_blocking = False
    self.update()

  # returns the RGB image currently displayed by the viewer
  def get_image(self):
    converter = vtk.vtkWindowToImageFilter()
    converter.SetInput(self.render_window)
    converter.ReadFrontBufferOff()
    converter.Update()
    im = vtk_to_numpy(converter.GetOutput().GetPointData().GetScalars())
    return np.flipud(im.reshape(self.height, self.width, im.shape[-1]))

  # returns: points in an Nx3 numpy array
  def get_points(self, mesh_id):
    if not mesh_id in self._mesh_actors:
      return None

    return vtk_to_numpy(self._get_mesh_polydata(mesh_id).GetPoints().GetData())

  # returns: point colors in an Nx3 numpy array
  def get_point_colors(self, mesh_id):
    if not mesh_id in self._mesh_actors:
      return None

    colors = self._get_mesh_polydata(mesh_id).GetPointData().GetScalars()
    if colors is not None:
      colors = vtk_to_numpy(colors)

    return colors

  def get_point_normals(self, mesh_id):
    if not mesh_id in self._mesh_actors:
      return None

    polyData = self._get_mesh_polydata(mesh_id)
    # TODO: how to deal with transformed meshes? Currently recomputing each time
    #normals = polyData.GetPointData().GetNormals()
    #if normals is None:
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polyData)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOff()
    normal_generator.SplittingOff()
    normal_generator.Update()
    normals = normal_generator.GetOutput().GetPointData().GetNormals()
    #polyData.GetPointData().SetNormals(normals)

    return vtk_to_numpy(normals)

  # returns the current camera pose
  # R: world-to-camera rotation
  # t: world-to-camera translation
  #    R follows a right-hand coordinate system viewing down the +Z axis,
  #    with the +Y axis going down
  def get_pose(self):
    t = np.array(self._camera.GetPosition())

    # need to convert R to the correct orientation (180 deg around x axis)
    R = _vtkMatrix4x4_to_numpy(
      self._camera.GetModelViewTransformMatrix())[:3,:3]
    R_corrective = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1.)))
    R = R_corrective.dot(R).T

    return R, t
    
  # return (camera-centered) z values for every pixel in the image
  # inceased_precision: If True, uses a (more expensive) multi-pass technique to
  #                     obtain more accurate depth values
  # TODO: haven't actually implemented a multi-pass thing; right now, it just
  # figures out the range of depth values and re-runs depth estimation only on
  # this clipping range
  def get_z_values(self, increased_precision=False):
    z = self._get_z_values()

    if increased_precision:
      near_original, far_original = self.near_clipping, self.far_clipping

      mask = (z < far_original) # TODO: figure out better approach

      if np.count_nonzero(mask) == 0:
        return z # nothing rendered, or everything too far away

      zmin, zmax = np.min(z), np.max(z[mask])

      if zmax == zmin:
        zmin *= 0.5 # TODO: everything is too close, it seems...

      self.set_clipping_range(zmin, zmax)

      z[mask] = self._get_z_values()[mask]

      # reset the camera clipping range
      self.set_clipping_range(near_original, far_original)

    return z

    #if update_clipping_range:
    #  min_z, max_z = np.inf, 0
    #  for actor in self._mesh_actors.itervalues():
    #    points = t[np.newaxis,:] + vtk_to_numpy(
    #        actor.GetMapper().GetInput().GetPoints().GetData()).dot(R.T)
    #    min_z = min(min_z, np.min(points[:,2]))
    #    max_z = max(max_z, np.max(points[:,2]))
    #
    #  min_z = max(min_z, 0.001)
    #  max_z = max(max_z, min_z + 0.1)
    #  print min_z, max_z
    #  #self._camera.SetClippingRange(min_z, max_z)
    #  self.interactor().SetClippingRange(min_z, max_z)

  # returns:
  #   mesh_id:  ID for the added mesh
  def load_obj(self, obj_file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return self._add_mesh_actor(actor)

  # returns:
  #   mesh_id:  ID for the added mesh
  def load_ply(self, ply_file):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(ply_file)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return self._add_mesh_actor(actor)
    
  # points3D:   (Mx)Nx3 array of (x,y,z) coordinates
  # colors:     (Mx)Nx3 array of (R,G,B) values [0 to 255]
  # mesh:       False to just show points (points and colors are Nx3 arrays);
  #             True to form a mesh (MxNx3), e.g. for SFS surface visualization
  #
  # returns:
  #   mesh_id:  ID for the added mesh
  def load_point_cloud(self, points, colors=None, mesh=False):
    # first, generate quad mesh, if desired
    if mesh:
      mesh = vtk.vtkCellArray()
      if len(points.shape) != 3 or points.shape[2] != 3:
        raise Exception('For the meshing to work, the points array must have ' +
          'the form MxNx3')

      for i in xrange(points.shape[0] - 1):
        offset = i * points.shape[1]
        for j in xrange(points.shape[1] - 1):
          quad = vtk.vtkQuad()
          pids = quad.GetPointIds()
          pids.SetNumberOfIds(4)
          idx = offset + j
          pids.SetId(0, idx)
          pids.SetId(1, idx + 1)
          pids.SetId(2, idx + points.shape[1] + 1)
          pids.SetId(3, idx + points.shape[1])
          mesh.InsertNextCell(quad)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(self._convert_points_array(points))

    if mesh is not False:
      poly_data.SetPolys(mesh)
    else: # no mesh data; only show vertices
      vertices = vtk.vtkCellArray()
      vertices.InsertNextCell(points.shape[0], np.arange(points.shape[0]))
      poly_data.SetVerts(vertices)

    # Visualize
    actor = self._actor_from_poly_data(poly_data)
    actor.GetProperty().SetPointSize(self.point_size)

    if colors is not None:
      self._add_color_to_actor(actor, colors)

    return self._add_mesh_actor(actor)
     
  # move the crosshair to the point x, y, z
  # TODO: support rotation of the crosshair
  def move_crosshair(self, x, y, z, s=1.):
    poly_data = self._crosshair_actor.GetMapper().GetInput()

    points = self._generate_crosshair_points(x, y, z, s)
    poly_data.SetPoints(self._convert_points_array(points))
    self.update()

  def remove_landmark(self, pos):
    actor = self._landmark_actors.pop(pos)
    self._renderer.RemoveActor(actor)
    self.update()

  # color: RGB background color with values in [0, 1]
  def set_background_color(self, *color):
    self._renderer.SetBackground(*color)

  # change the rendering to match a given pinhole camera model
  # any parameters unspecified are left unchanged
  def set_camera_params(self, width=None, height=None,
                        fx=None, fy=None, cx=None, cy=None):

    # Build a perspective transformation matrix from intrinsic camera parameters
    # see: http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    #
    # Here's an explanation of all that's going on:
    # There are two key differences between OpenGL and the camera coordinates
    # we know and love:
    # 1) OpenGL image coordinates start from the bottom left, rather than the
    #    top left. OpenGL is still a RHS, but the camera views down the -z axis,
    #    and thus the image plane is at z=-1.
    # 2) Image coordinates lie in Normalized Device Coordinates, that is,
    #    [-1,1] x [-1,1], rather than [0,w-1] x [0, h-1].
    #
    # Because of this, several changes to camera calibration matrix must first
    # be made:
    # 1) cy := -cy (to account for the change in origin)
    # 2) The third column of the camera matrix K is multiplied by -1 (to account
    #    for the image plane being at z=-1.
    # 3) cx' = 2 * cx / (w - 1) - 1 (and similarly for cy') for NDC purposes.
    # 4) A new focal length, fx', needs to be computed for NDC; consider a
    #    normalized point xhat (which is the same, regardless of whether we're
    #    using pixel coordinates or NDC):
    #      xhat = (x - cx) / fx = ((2*x/(w-1) - 1) - (2*cx/(w-1) - 1)) / fx'
    #    Then,
    #      fx' = fx * 2 / (w-1)
    #    and similarly for fy'.
    #
    # Finally, a third row/final column needs to be added to the new camera
    # matrix, to account for mapping of z values. More details can be found
    # above, but in short, the resulting perspective transformation matrix is:
    #       [fx'  0  -cx' 0]
    #   P = [ 0  fy'  cy' 0]
    #       [ 0   0    A  B]
    #       [ 0   0   -1  0],
    #
    # where, if the camera clipping planes are near and far (both negative),
    #   A = near + far
    #   B = near * far
    #
    # These values are programatically determined by
    #   M = vtkCamera.GetProjectionMatrix(aspect_ratio, near, far)
    # for default camera parameters. What we'll do is build our desired matrix P
    # and then calculate a "user" matrix U such that UM = P.

    if width is not None:
      self.width = int(width)
    if height is not None:
      self.height= int(height)

    # focal length in normalized device coordinates
    if fx is not None:
      self._fx = 2. * fx / (self.width - 1)
    if fy is not None:
      self._fy = 2. * fy / (self.height - 1)

    # principal point normalized to [-1,1]
    # these values are actually -cx and -cy (which was already negated), since
    # we multiply by -1 in the projection matrix (to account for the image plane
    # lying at z=-1)
    if cx is not None:
      self._cx = -((2. * cx) / (self.width - 1) - 1)
    if cy is not None:
      self._cy = (2. * cy) / (self.height - 1) - 1

    # actual perspective transformation building is done in
    # _update_camera_params(), as the near/far planes can be altered, too
    self._update_camera_params()

  def set_clipping_range(self, near_clipping=None, far_clipping=None):
    # TODO see also: http://www.vtk.org/Wiki/VTK/Examples/Cxx/Meshes/SolidClip
    if near_clipping is not None:
      self.near_clipping = near_clipping
    if far_clipping is not None:
      self.far_clipping = far_clipping

    self._camera.SetClippingRange(self.near_clipping, self.far_clipping)
    self._update_camera_params()

  def set_point_size(self, point_size):
    for actor in self._mesh_actors.itervalues():
      actor.GetProperty().SetPointSize(self.point_size)
    self.update()            
    
  # R: world-to-camera-coordinates rotation
  #    R should follow a right-hand coordinate system viewing down the +Z axis,
  #    with the +Y axis going down
  # t: world-to-camera-coordinates translation
  def set_pose(self, R=np.eye(3), t=np.array([0., 0., -1.])):
    self._camera.SetPosition(t)
    self._camera.SetFocalPoint(t + R.dot([0., 0., 1.]))
    self._camera.SetViewUp(R.dot([0., -1., 0.]))     
    self.update()

  def toggle_crosshair(self, visible=None):
    self._crosshair_actor.SetVisibility(
      not self._crosshair_actor.GetVisibility() if visible is None else visible)
    self.update()            

  def toggle_lighting(self):
    self._lighting_on = not self._lighting_on
    for actor in self._mesh_actors.itervalues():
      actor.GetProperty().SetLighting(self._lighting_on)
    self.update()            

  # visible: set to True or False to explicitly set the visibility
  def toggle_mesh_visibility(self, mesh_id, visible=None):
    if mesh_id in self._mesh_actors:
      actor = self._mesh_actors[mesh_id]
      actor.SetVisibility(
        not actor.GetVisibility() if visible is None else visible)
      self.update()

  #
  # update commands
  #

  def update(self):
    if not self._render_blocking:
      self.render_window.Render()

  # Provide a new (Nx3 or MxNx3) array of 3D points (optionally with color), and
  # replace the point set of the given mesh with the new points. This is more
  # efficient than creating an entirely new mesh and deleting the old one.
  def update_points(self, mesh_id, points3D=None, colors=None):
    if not mesh_id in self._mesh_actors:
      return

    if points3D is not None:
      # directly reference the point cloud data from numpy arrays
      points3D = np.ascontiguousarray(
        points3D.reshape(-1, 3).astype(np.double, subok=True, copy=False))

      points_array = numpy_to_vtk(points3D)
      # to avoid points3D going out of scope
      setattr(points_array, '_np_points_array', points3D)

      points = vtk.vtkPoints()
      points.SetData(points_array)
      points.SetDataTypeToDouble()
        
      poly_data = self._get_mesh_polydata(mesh_id)
      poly_data.SetPoints(points)

    if colors is not None:
      self._add_color_to_actor(self._mesh_actors[mesh_id], colors)
     
    self.update()
     
