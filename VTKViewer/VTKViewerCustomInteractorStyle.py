import vtk

from collections import defaultdict

# mapping event names to On<Call>() attributes
# this is necessary because VTK doesn't standardize its event names
# for instance, OnLeftButtonDown has an event name of 'LeftButtonPressEvent'
# TODO: these events are only for TrackballCamera; there may be other events we
#       could add
EVENTS = dict((
  ('CharEvent', 'OnChar'),
  ('ConfigureEvent', 'OnConfigure'),
  ('EnterEvent', 'OnEnter'),
  ('ExposeEvent', 'OnExpose'),
  ('KeyPressEvent', 'OnKeyPress'),
  ('KeyReleaseEvent', 'OnKeyRelease'),
  ('LeaveEvent', 'OnLeave'),
  ('LeftButtonPressEvent', 'OnLeftButtonDown'),
  ('LeftButtonReleaseEvent', 'OnLeftButtonUp'),
  ('MiddleButtonPressEvent', 'OnMiddleButtonDown'),
  ('MiddleButtonReleaseEvent', 'OnMiddleButtonUp'),
  ('MouseMoveEvent', 'OnMouseMove'),
  ('MouseWheelBackwardEvent', 'OnMouseWheelBackward'),
  ('MouseWheelForwardEvent', 'OnMouseWheelForward'),
  ('RightButtonPressEvent', 'OnRightButtonDown'),
  ('RightButtonReleaseEvent', 'OnRightButtonUp'),
  ('TimerEvent', 'OnTimer')))

# VTKViewerCustomInteractorStyle
#
# Creates a new VTKInteractorStyle based on a given base InteractorStyle
# (default is TrackballCamera). Events supporting (strict) combinations of the
# alt, control, and shift keys can be added/removed using the
# register_/deregister_event functions. The only events that are currently
# supported for registration are those that have an associated On<Event>
# function in the base InteractorStyle.
#
# inputs:
#   base: the base type of interactor style to use
def VTKViewerCustomInteractorStyle(base=vtk.vtkInteractorStyleTrackballCamera):
  class VTKViewerCustomInteractorStyle(base):
    def __init__(self):
      # suppressing events prevent other events from being fired
      # if no suppressing events are registered, all nonsuppressing events fire
      self._suppressing_events = defaultdict(list)
      self._nonsuppressing_events = defaultdict(list)

      for event_name, event_attr in EVENTS.iteritems():
        if hasattr(self, event_attr):
          self.AddObserver(event_name, self._local_callback)

    # custom callback by which we register events
    def _local_callback(self, obj, event_name):
      interactor = self.GetInteractor()

      # iterate through suppressing events; if any checks out, run it and return
      for properties, callback in self._suppressing_events[event_name]:
        if properties.anykey or (
            not (properties.alt  ^ interactor.GetAltKey()) and
            not (properties.ctrl ^ interactor.GetControlKey()) and
            not (properties.shft ^ interactor.GetShiftKey())):
          callback()
          return

      # no suppressing events occurred, so perform all remaining events
      for properties, callback in self._nonsuppressing_events[event_name]:
        if properties.anykey or (
            not (properties.alt  ^ interactor.GetAltKey()) and
            not (properties.ctrl ^ interactor.GetControlKey()) and
            not (properties.shft ^ interactor.GetShiftKey())):
          callback()

      # the original event for the base interactor style
      getattr(self, EVENTS[event_name])()

    # anykey: if True, the event fires regardless of modifier keys pressed
    # alt/ctrl/shft: modifier keys combination associated with this event
    # suppress: stop other events with this key combination from occurring
    def register_event(self, event_name, event_callback, anykey=False,
                       alt=False, ctrl=False, shft=False, suppress=False):
      event_properties = type(
        event_name, tuple(), dict(anykey=anykey, alt=alt, ctrl=ctrl, shft=shft))

      if suppress: # mark that this event stops all other events from firing
        self._suppressing_events[event_name].append(
            (event_properties, event_callback))
      else:
        self._nonsuppressing_events[event_name].append(
            (event_properties, event_callback))

    # remove *all* registered events of this type having the given callback
    def deregister_event(self, event_name, event_callback):
      self._suppressing_events[event_name] = filter(
        lambda e, f: f == event_callback, self._events[event_name])
      self._nonsuppressing_events[event_name] = filter(
        lambda e, f: f == event_callback, self._events[event_name])

  return VTKViewerCustomInteractorStyle()

