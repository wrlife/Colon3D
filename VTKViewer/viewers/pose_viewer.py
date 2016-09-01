# TODO: just putting this here, for now; needs to be fleshed out a little more,
# like MeshViewer

# also displays camera pose information below the viewer
# to update the text, use PoseViewer.text.set('text')
class PoseViewer(VTKViewerTk):
  def __init__(self, root=None, width=VTKViewerTk.DEFAULT_WIDTH,
               height=VTKViewerTk.DEFAULT_HEIGHT):
    VTKViewerTk.__init__(self, root, width, height)
    self._initialize_ui()

  def _initialize_ui(self):
    frame = Tk.Frame(self.root, width=self.width, height=50)
    self.text = Tk.Text(frame, state='disabled', height=2)
    self.text.bind('<1>', lambda event: self.text.focus_set()) # allows copy
    self.text.pack(fill=Tk.BOTH, expand=Tk.YES)
    frame.pack(before=self.interactor, side=Tk.BOTTOM, anchor=Tk.W)
    frame.pack_propagate(False)
    self.update_pose_text()

    # disable zooming using the mouse wheel
    self.interactor_style.register_event('MouseWheelForwardEvent',
        lambda: None, suppress=True)
    self.interactor_style.register_event('MouseWheelBackwardEvent',
        lambda: None, suppress=True)
    self.interactor_style.register_event(
        'MouseMoveEvent', self.update_pose_text, anykey=True)

  def update_pose_text(self):
    R, t = self.get_pose()
    q = Quaternion.FromR(R).q
    self.text.config(state=Tk.NORMAL)
    self.text.delete(1.0, Tk.END)
    self.text.insert(Tk.END,
      'qvec: [ %0.3f %0.3f %0.3f %0.3f ]\ntvec: [ %0.3f %0.3f %0.3f ]' %
      tuple(q.tolist() + t.tolist()))
    self.text.config(state=Tk.DISABLED)


