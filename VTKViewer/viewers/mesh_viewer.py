from .. import VTKViewerTk

import Tkinter as Tk

class TkMeshList:
  def __init__(self, root):
    self.frame = Tk.Frame(master=root)
    title = Tk.Label(self.frame, text='Mesh List', width=20)
    title.grid(row=0)
    separator = Tk.Frame(self.frame, bg='black', height=1)
    separator.grid(row=1, padx=2, pady=1, sticky=Tk.W+Tk.E)
    self.frame.pack(side=Tk.RIGHT, anchor=Tk.N, fill=None, expand=False)

    self.checkbuttons = dict()

  def add_entry(self, mesh_id, mesh_name, onchange_callback):
    cb = Tk.Checkbutton(self.frame, text=mesh_name, command=onchange_callback)
    cb.select()
    cb.grid(row=len(self.checkbuttons) + 2, sticky=Tk.W)
    self.checkbuttons[mesh_id] = cb

  def remove_entry(self, mesh_id):
    if mesh_id in self.checkbuttons:
      self.checkbuttons[mesh_id].destroy()

class MeshViewer(VTKViewerTk):
  def __init__(self, root=None, width=VTKViewerTk.DEFAULT_WIDTH,
               height=VTKViewerTk.DEFAULT_HEIGHT):
    VTKViewerTk.__init__(self, root, width, height)
    self._initialize_ui()

  def _initialize_ui(self):
    crosshair_button = Tk.Button(self.root, text='Toggle Crosshair',
      command=self.toggle_crosshair)
    crosshair_button.pack()
    light_button = Tk.Button(self.root, text='Toggle Light',
      command=self.toggle_lighting)
    light_button.pack()
    self.mesh_list = TkMeshList(self.root)

  #
  # public functions
  #

  def clone_mesh(self, mesh_id, new_name):
    mesh_id = VTKViewerTk.clone_mesh(self, mesh_id)
    self.mesh_list.add_entry(mesh_id, new_name,
      lambda: self.toggle_mesh_visibility(mesh_id))
    return mesh_id

  def delete_mesh(self, mesh_id):
    VTKViewerTk.delete_mesh(self, mesh_id)
    self.mesh_list.remove_entry(mesh_id)

  def load_obj(self, filename):
    mesh_id = VTKViewerTk.load_obj(self, filename)
    self.mesh_list.add_entry(mesh_id, filename,
      lambda: self.toggle_mesh_visibility(mesh_id))
    return mesh_id

  def load_ply(self, filename):
    mesh_id = VTKViewerTk.load_ply(self, filename)
    self.mesh_list.add_entry(mesh_id, filename,
      lambda: self.toggle_mesh_visibility(mesh_id))
    return mesh_id

  def load_point_cloud(self, mesh_name, points, colors=None, mesh=False):
    mesh_id = VTKViewerTk.load_point_cloud(self, points, colors, mesh)
    self.mesh_list.add_entry(mesh_id, mesh_name,
      lambda: self.toggle_mesh_visibility(mesh_id))
    return mesh_id

