# Author: True Price <jtprice at cs.unc.edu>

import multiprocessing
import numpy as np

import sys
if sys.version_info[0] < 3:
  import Tkinter as Tk
else:
  import tkinter as Tk

from VTKViewer import VTKViewer
from vtkLoadPythonTkWidgets import vtkLoadPythonTkWidgets
from vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from VTKViewerCustomInteractorStyle import VTKViewerCustomInteractorStyle

class VTKViewerTk(VTKViewer):
  DEFAULT_WIDTH = 800
  DEFAULT_HEIGHT = 800

  POLL_RATE = 100 # poll for background process messages every 100ms

  def __init__(self, root=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    if root is None:
      root = Tk.Tk()    

      def _quit(*args):       
        root.quit()     # stops main loop
        root.destroy()  # this is necessary on Windows to prevent
                      # "Fatal Python Error: PyEval_RestoreThread: NULL tstate"

      root.protocol('WM_DELETE_WINDOW', _quit)
      root.bind('q', _quit) # q key will exit the program

    self.root = root

    self.interactor_style = VTKViewerCustomInteractorStyle()
    self.interactor_style.register_event(
      'MouseWheelForwardEvent',
      self._action_increase_point_size,
      ctrl=True, suppress=True)
    self.interactor_style.register_event(
      'MouseWheelBackwardEvent',
      self._action_decrease_point_size,
      ctrl=True, suppress=True)
    # TODO: need to get this working, as it's not compatible with setting the
    # camera parameters, etc.
    #self.interactor_style.register_event(
    #  'MouseWheelForwardEvent', self._action_increase_near_clipping, shft=True)
    #self.interactor_style.register_event(
    #  'MouseWheelBackwardEvent', self._action_decrease_near_clipping, shft=True)

    self.interactor = vtkTkRenderWindowInteractor(self.root, width=width,
                                                  height=height)
    self.interactor.SetInteractorStyle(self.interactor_style)
    self.interactor.pack(side=Tk.LEFT, fill=None, expand=False)

    render_window = self.interactor.GetRenderWindow()
    render_window.SetOffScreenRendering(False)

    # now that we have everything we need, go ahead and set up the base class
    VTKViewer.__init__(self, width, height, render_window)
    del self.render_window # a hack for compatibility; see __getattr__ below

    self.interactor.Initialize()
    self.interactor.Start()

  def __getattr__(self, attr):
    # because the tk part of vtkTkRenderWidget must have
    # the only remaining reference to the RenderWindow when
    # it is destroyed, we can't actually store the RenderWindow
    # as an attribute but instead have to get it from the tk-side
    if attr == 'render_window':
      return self.interactor.GetRenderWindow()
    raise AttributeError, self.__class__.__name__ + \
          " has no attribute named " + attr

  #
  # general private functions
  #

  def _action_decrease_near_clipping(self):
    if self.near_clipping > self.MIN_NEAR_CLIPPING:
      self.near_clipping /= self.NEAR_CLIPPING_SCROLL_FACTOR
      self.set_near_clipping(self.near_clipping)

  def _action_increase_near_clipping(self):
    if self.near_clipping < self.MAX_NEAR_CLIPPING:
      self.near_clipping *= self.NEAR_CLIPPING_SCROLL_FACTOR
      self.set_near_clipping(self.near_clipping)

  def _action_decrease_point_size(self):
    if self.point_size > self.MIN_POINT_SIZE:
      self.point_size /= self.POINT_SIZE_SCROLL_FACTOR
      self.set_point_size(self.point_size)

  def _action_increase_point_size(self):
    if self.point_size < self.MAX_POINT_SIZE:
      self.point_size *= self.POINT_SIZE_SCROLL_FACTOR
      self.set_point_size(self.point_size)

  # for background processes that have been registered, this allows a polling
  # procedure for the parent process (i.e. the window) to execute a callback
  # whenever the child process posts a message
  def _poll_background_process(self, parent_conn, child_proc, poll_callback):
    if parent_conn.poll():
      while parent_conn.poll(): # in case multiple messages to receive
        poll_callback(parent_conn.recv())

    if child_proc.is_alive(): # re-register polling
      self.after(VTKViewerTk.POLL_RATE,
                 self._poll_background_process,
                 parent_conn, child_proc, poll_callback)

  #
  # public functions
  #

  # equivalent to self.root.after(...)
  # this registers Tkinter to execute callback() with the given arguments after
  # some delay (after begin() has been called)
  def after(self, delay_ms, callback=None, *args):
    self.root.after(delay_ms, callback, *args)

  def begin(self):
    self.root.mainloop()

  # run the function specified by func in a background process; the process will
  # start after self.begin() has been called, or immediately if that function
  # has already been called
  #
  # inputs:
  #   poll_callback: if specified, a communication channel will be established
  #     between the child and parent processes; the parent process will poll for
  #     messages from the background process and will execute the callback,
  #     passing in contents of the message, whenever there is a message
  #
  # returns:
  #   parent_conn: connection for sending messages from the parent process to
  #     its child; messages can be posted using parent_conn.send(<object>)
  #   child_conn: connection for sending messages from the child process to its
  #     parent; messages can be posted using child_conn.send(<object>)
  def run_background_process(self, func, args=tuple(), kwargs=dict(),
                             poll_callback=None):
    child_proc = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    child_proc.daemon = True
    self.after(0, child_proc.start)

    parent_conn, child_conn = multiprocessing.Pipe()
  
    if poll_callback is not None:
      self.after(0, self._poll_background_process,
                 parent_conn, child_proc, poll_callback)

    return parent_conn, child_conn

###
##  def raycastFromClick( self, mouseLoc ):
##    pSource = #cameraloc
##    pTarget = #mouseloc + some amt
##    obbTree = vtk.vtkOBBTree()
##    obbTree.SetDataSet( self._image )
##    obbTree.BuildLocator()
##    pointsVTKintersection = vtk.vtkPoints()
##    code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)
##    if code:
##        pointsVTKIntersectionData = pointsVTKintersection.GetData()
##        noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
##        pointsIntersection = []
##        for idx in range(noPointsVTKIntersection):
##            _tup = pointsVTKIntersectionData.GetTuple3(idx)
##            pointsIntersection.append(_tup)

