import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.cm as cm
import oop_plot_data as mc
import scipy.optimize as optim

from gui_page import *

# FTIR data
class Page1(Page):
  def __init__(self, *args, **kwargs):
    Page.__init__(self, *args, **kwargs)
