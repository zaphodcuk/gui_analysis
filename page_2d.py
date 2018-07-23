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
import gui_config as cfg
import pathlib as pal

md_path = pal.Path(cfg.md_path)

f2 = Figure(figsize=(5,5), dpi=100)
a2 = f2.add_subplot(111)
f3 = Figure(figsize=(4,3), dpi=100)
a3 = f3.add_subplot(111)

class Page3(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.negative = tk.IntVar()
        self.calib = tk.IntVar()
        self.data = None
        l2d = tk.Button(self,
                        text="Load 2D",
                        command=self.load_2d).grid(column=1,
                                                   row=1)
        tk.Checkbutton(self, 
                       text="Subtract negative",
                       variable=self.negative,
                       command=self.subtract_negative).grid(column=2,
                                                            row=1)

        tk.Checkbutton(self, 
                       text="probe calibration",
                       variable=self.calib,
                       command=self.probe_calib).grid(column=5,
                                                            row=1)

        
        self.canvas = FigureCanvasTkAgg(f2, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=1,
                                         row=2,
                                         columnspan=10,
                                         rowspan=10)

    
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=21,column=1,columnspan=10)
        toolbar = NavigationToolbar2TkAgg( self.canvas, toolbar_frame )
        toolbar.update()
        self.canvas._tkcanvas.grid()

        self.delay_list = tk.Listbox(self)
        self.delay_list.grid(column=1,row=2)
        self.delay_list.bind('<<ListboxSelect>>', self.update_idx_kin)
        self.delay_list.activate(0)
        self.idx = 0

        self.canvas2 = FigureCanvasTkAgg(f3, self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(column=14,
                                         row=2,
                                         columnspan=10,
                                         rowspan=10)

    
        toolbar2_frame = tk.Frame(self)
        toolbar2_frame.grid(row=21,column=4,columnspan=10)
        toolbar2 = NavigationToolbar2TkAgg( self.canvas2, toolbar2_frame )
        toolbar2.update()
        self.canvas2._tkcanvas.grid()

        self.slide_hor = tk.Scale(self,from_=0, to=31, command=self.update_idx_kin)
        self.slide_hor.grid(column=11, row=2,in_=self)
        self.slide_hor.set(15)
        self.idx_hor = self.slide_hor.get()
        
        

    def refreshFigure(self):
        self.canvas.draw()
        self.canvas2.draw()

    def update_idx_kin(self,event):
        if self.data:
            self.idx_kin = self.delay_list.curselection()[0]
            self.plot_2d()
        else:
            pass

    def subtract_negative(self):
        self.data.subtract_first_spectrum()
        self.plot_2d()

        

    def load_2d(self):
        if cfg.debug == 1:
            self.data = mc.MD(md_path)
        else:
            self.data = mc.MD()
        for i in self.data.t:
            self.delay_list.insert(tk.END,i)
        self.data.pre_phase()
        self.data.apodize()
        self.data.padding(self.data.avinterf, self.data.avsignal)
        self.data.phase(self.data.avsignal, self.data.avinterf, pump_corr=0)
        print('data loaded')
        self.plot_2d()


    def plot_2d(self):
        a3.clear()
        offset =0
        ib = np.argmin( abs(self.data.ax_pump - self.data.w[0]) )
        ie = np.argmin( abs(self.data.ax_pump - self.data.w[-1]) ) +1
        if self.negative.get() == 0:
            data2plot = self.data.fft_sig
        else:
            data2plot = self.data.data_sub_neg

        levels = mc.cont_range(np.real(data2plot[ib:ie,:,self.idx]),
                               off=offset,ncont=41, mode = 'auto')
        X, Y = np.meshgrid(self.data.ax_pump[ib:ie], self.data.w)
        a2.contourf(X,Y,
                    np.transpose(data2plot[ib:ie,:,self.idx])-offset,
                    levels,
                    cmap=cm.seismic)


        # pp proj
        self.pp_proj = np.zeros((self.data.lt,32))
        for i in range(self.data.lt):
            self.pp_proj[i,:] = np.sum(np.real(data2plot[ib:ie,:,i]),axis=0)
        a3.plot(self.data.w, self.pp_proj[self.idx,:])
        self.refreshFigure()

    def probe_calib(self):
        if self.negative.get() == 1:
            raise ValueError('Not possible to calibrate 2D after negative time subtraction')
        if self.calib.get() == 1:
            ib = np.argmin( abs(self.data.ax_pump - self.data.w[0]) ) -5
            ie = np.argmin( abs(self.data.ax_pump - self.data.w[-1]) ) +5
            max_signal = []
            for i in range(32):
                max_signal.append(np.argmax(abs(self.data.fft_sig[ib:ie,i,0])) )
                self.data.w[i] = self.data.ax_pump[ib+max_signal[-1]]
        else:# this is wrong if there is wavelengts
            self.data.w = np.arange(32)
        self.plot_2d()
