import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import *

import os
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from page_pp import *
from page_ftir import *
from page_2d import *
#### settings part ####

path3 = pal.Path('/home/dpalec/Documents/UZH/Data/2D/Water_h/20180528/2D_TiO2_Pt_KI_meoh_UV7_203411')
path4 = pal.Path('/home/david/Dokumenty/UZH/Data/2D/Water_h/20180528/2D_TiO2_Pt_KI_meoh_UV7_203411')

md_path = path4
format_data = 'lab1'

class MainApp(ttk.Notebook):
  def __init__(self, *args, **kwargs):
    ttk.Notebook.__init__(self, *args, **kwargs)
    p1 = Page1(self)
    p2 = Page2(self)
    p3 = Page3(self)

    self.add(p1,text='FTIR')
    self.add(p2,text='PP')
    self.add(p3,text='2D')
    but_close = tk.Button(text='quit', command=root.destroy).grid(column=15, row=1, in_=root)

if __name__ == "__main__":
  root = tk.Tk()
  main = MainApp(root).grid()
  root.wm_geometry("1500x700")
  root.mainloop()
