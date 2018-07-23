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
import pathlib as pal
import gui_config as cfg

from gui_page import *
from gui_fitting import *

f = Figure(figsize=(6,4), dpi=100)
a = f.add_subplot(111)
ff = Figure(figsize=(4,3), dpi=100)
aa = ff.add_subplot(111)
f4 = Figure(figsize=(4,3), dpi=100)
a4 = f4.add_subplot(111)

f01 = Figure(figsize=(4,3), dpi=100)
a01 = f01.add_subplot(111)
f02 = Figure(figsize=(4,3), dpi=100)
a02 = f02.add_subplot(111)

pp_path = pal.Path(cfg.pp_path)

class Page2(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.data = None
        self.t = None
        self.scale = tk.IntVar()
        self.sub_mean = tk.IntVar()
        self.fit_constant = tk.IntVar()
        self.dformat = tk.StringVar()
        self.dformat.set(cfg.pp_format) # default value from the config file.

        self.options = tk.Frame(self, relief='groove',borderwidth=2)
        self.options.grid(row=0,column = 0)
        lpp = tk.Button(self,
                        text="Load PP",
                        command=self.load_pp).grid(column=0,
                                                   row=0,
                                                   in_=self.options)

        # plot options
        tk.Checkbutton(self, 
                text="Log scale",
                variable=self.scale,
                command=self.refreshFigure).grid(column=0,
                                                 row=1,
                                                 in_=self.options)
        tk.Checkbutton(self, 
                text="Remove mean",
                variable=self.sub_mean,
                command=self.subtract_mean).grid(column=0,
                                                 row=2,
                                                 in_=self.options)

        self.format_menu = tk.OptionMenu(self, self.dformat, "lab1", "lab4", "cbr")
        self.format_menu.grid(column = 1, row = 0, in_=self.options)
        
        self.cplot = tk.Frame(self, relief='groove',borderwidth=2)
        self.cplot.grid(row=1,column = 0)
        
        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0,
                                        row=1, in_=self.cplot)

    
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=2,column=0,sticky=tk.W)
        toolbar = NavigationToolbar2TkAgg( self.canvas, toolbar_frame )
        toolbar.update()
        self.canvas._tkcanvas.grid()

        
        # I want to make it a notebook
        subframe = tk.Frame(self)
        subframe.grid(row=0,column=2,rowspan=5,sticky=tk.E)
        self.sub_pages = ttk.Notebook(self)#subpages(self)
        self.sub_pages.grid(column=1, row=0,in_=subframe)
        self.sPage1 = tk.Frame(self)
        self.sub_pages.add(self.sPage1, text='Cuts')
        self.sPage2 = tk.Frame(self)
        self.sub_pages.add(self.sPage2, text='DAS')


        #kinetics
        self.canvas2 = FigureCanvasTkAgg(ff, self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(column=2,
                                          row=0,
                                          in_=self.sPage1)
        toolbar2_frame = tk.Frame(self)
        toolbar2_frame.grid(row=1,column=2, sticky=tk.W)
        toolbar2 = NavigationToolbar2TkAgg( self.canvas2, toolbar2_frame )
        toolbar2.update()
        self.canvas2._tkcanvas.grid()

        ## TODO: slider should show wavenumbers.
        self.slide_kin = tk.Scale(self,from_=0, to=31, command=self.update_idx_kin)
        self.slide_kin.grid(column=3, row=0,in_=self.sPage1)
        self.slide_kin.set(15)
        self.idx_kin = self.slide_kin.get()

        self.slide_sp = tk.Scale(self,from_=0, to=250, command=self.update_idx_sp)
        self.slide_sp.grid(column=3, row=1,in_=self.sPage1)
        self.slide_sp.set(15)
        self.idx_sp = self.slide_sp.get()

        # fitting kinetics
        self.fields = [['a1', 't1', 'c'],
                       ['a1', 't1', 'a2','t2','c'],
                       ['a1', 't1', 'a2','t2', 'a3', 't3', 'c']]
        fields2 = 't0','Nexp',
        self.fit_frame = tk.Frame(self, relief='groove',borderwidth=2)
        self.fit_frame.grid(column=4,row=0,in_=self.sPage1)
        self.ents = self.create_fit_fields(fields2, self.fit_frame)

        # spectra
        self.canvas3 = FigureCanvasTkAgg(f4, self)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().grid(column=2,
                                          row=1,
                                          in_=self.sPage1)

        toolbar3_frame = tk.Frame(self)
        toolbar3_frame.grid(row=2,column=2,in_=self.sPage1,sticky=tk.W)
        toolbar3 = NavigationToolbar2TkAgg( self.canvas3, toolbar3_frame )
        toolbar3.update()
        self.canvas3._tkcanvas.grid()

        # DAS components
        self.canvas4 = FigureCanvasTkAgg(f01, self)
        self.canvas4.draw()
        self.canvas4.get_tk_widget().grid(column=0,
                                          row=0,
                                          in_=self.sPage2)
        toolbar4_frame = tk.Frame(self)
        toolbar4_frame.grid(row=0,column=0, in_=self.sPage2,sticky=tk.W)
        toolbar4 = NavigationToolbar2TkAgg( self.canvas4, toolbar4_frame )
        toolbar4.update()
        self.canvas4._tkcanvas.grid()
        
        self.fit_frame = tk.Frame(self, relief='groove',borderwidth=2)
        self.fit_frame.grid(column=4,row=0, in_=self.sPage2)
        self.ents_das = self.create_fit_fields_DAS(fields2, self.fit_frame)

        # residuals
        self.canvas5 = FigureCanvasTkAgg(f02, self)
        self.canvas5.draw()
        self.canvas5.get_tk_widget().grid(column=0,
                                          row=1,
                                          in_=self.sPage2)
        toolbar5_frame = tk.Frame(self, height=2, width=10)
        toolbar5_frame.grid(row=1,column=0, in_=self.sPage2,sticky=tk.W)
        toolbar5 = NavigationToolbar2TkAgg( self.canvas5, toolbar5_frame )
        toolbar5.update()
        self.canvas5._tkcanvas.grid()
        # self.a_list = []
        # self.a_list += [1]*32
        # self.a_list += [5]*32
        # self.t_list = [3,30]


    def create_fit_fields(self, fields, inframe):
        entries = []
        row = 0
        for field in fields:
            lab = tk.Label(self, width=5, text=field)
            ent = tk.Entry(self,width=5)
            ent.insert(row, '1')
            

            lab.grid(row = row, column=0, in_=inframe)
            ent.grid(row = row, column=1, in_=inframe)

            entries.append((field, ent))
            row += 1
        self.fit_text = tk.Text(self, width = 10, height = 7)
        self.fit_text.grid(row=2, column = 0, in_=inframe)

        tk.Checkbutton(self,
                    text='fit constant',
                    variable = self.fit_constant).grid(row=3,
                                                       column=1,
                                                       in_=inframe)
        return entries
    
        
    def refreshFigure(self):
        if self.scale.get() == 0:
            ax = self.canvas.figure.axes[0]
            ax.set_xscale("linear")
            #for kinetics
            ax2 = self.canvas2.figure.axes[0]
            ax2.set_xscale("linear")
        elif self.scale.get() == 1:
            ax = self.canvas.figure.axes[0]
            ax.set_xlim(1e-1)
            ax.set_xscale("log", nonposx='clip')
            # for kinetics
            ax2 = self.canvas2.figure.axes[0]
            ax2.set_xlim(1e-1)
            ax2.set_xscale("log", nonposx='clip')
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()


    def load_pp(self):
        if cfg.debug == 1:
            self.data = mc.PP(str(pp_path), format_data = self.dformat.get())
        else:
            self.data = mc.PP(format_data = self.dformat.get())
        self.plot_pp()
        print('data loaded')

    def subtract_mean(self):
        self.data.remove_mean()
        self.plot_pp()

    def update_idx_kin(self,value):
        if self.data:
            self.idx_kin = self.slide_kin.get()
            self.plot_pp()
            self.fit_kin(self.ents)            
        else:
            pass

    def update_idx_sp(self,value):
        if self.data:
#            a4.clear()
            self.idx_sp = self.slide_sp.get()
            self.plot_pp()
#            a4.plot(self.data.w,self.data2plot[self.idx_sp,:])
#            self.refreshFigure()
        else:
            pass

    def fit_kin(self,ents):
        #decide how many inputs and exponentials I have
        fit_init = get_entries(ents)
        fit = fit_exponentials(self.data2plot,
                         self.data.t,
                         self.data.w,
                         idx = self.idx_kin,
                         nexp = fit_init[-1],
                         tinit = fit_init[-2],
                         log = self.scale.get())
        aa.plot(self.data.t[fit_init[-2]:],
                mc.func(self.data.t[fit_init[-2]:],
                        fit[0][0],fit_init[-1]))
        self.refreshFigure()
        self.fit_text.delete('1.0',tk.END)
        for i in range(len(fit[0][0])):
            self.fit_text.insert(tk.INSERT,self.fields[int(fit_init[-1])-1][i] +
                                 ': ' + str(np.round(fit[0][0][i],2)) + '\n')

    def das(self,ents):
        fit_init = get_entries(ents)
        nexp = int(fit_init[-1])
        a_list = []
        i_list = [5,7,10,-1]
        t_list = [3,30,50,100]
        tau_list = t_list[:nexp]
        for i in range(nexp):
            a_list.append([i_list[i]]*len(self.data.w))

        print('data shape',self.data2plot.shape)
        print(len(self.data.t), nexp)
        print(a_list)
        print(tau_list)
        fit = fit_das(self.data2plot, self.data.t, nexp,a_list,tau_list,t0 = fit_init[-2])
        print('final fit',fit)
        self.plot_das_components(fit,nexp,t0=fit_init[-2])
        return fit

    def plot_das_components(self,fit_result,nexp,t0):
        a02 = f02.add_subplot(111)
        #self.canvas.draw()
        a01.clear()
        a02.clear()
        fit_a_list = []
        fit_t_list = []
        wlen = len(self.data.w)
        idx = np.argmin(abs(self.data.t-t0))
        for i in range(nexp):
            fit_a_list.append(fit_result[0][0][i*wlen:i*wlen+wlen])
            fit_t_list.append(fit_result[0][0][-nexp+i])
        result_fit = model_das(self.data.t,wlen,fit_a_list,fit_t_list,nexp)
        for i in range(nexp):
            if fit_t_list[i] > 10000:
                fit_t_list[i] = 'NaN'
                a01.plot(self.data.w,fit_a_list[i],label ='NaN')
            else:
                a01.plot(self.data.w,fit_a_list[i],
                         label = '%s' % float('%.2g' % fit_t_list[i]))#np.round(fit_t_list[i],2))
        a01.legend(fontsize=10)

        fit_res = self.data2plot[idx:,:]-result_fit.reshape(len(self.data.t),wlen)[idx:,:]
        levels = mc.cont_range(fit_res,
                                0, # offset
                                ncont=31)
        res = a02.contourf(fit_res,levels,cmap=cm.seismic)
        cb = f02.colorbar(res)
        self.canvas4.draw()
        self.canvas5.draw()
        f02.delaxes(f02.axes[1])

    def plot_pp(self):
        aa.clear()
        a4.clear()
        a.clear()
        if self.sub_mean.get() == 0:
            self.data2plot = self.data.avdata
        elif self.sub_mean.get() == 1:
            self.data2plot = self.data.data_wto_mean

        

        levels = mc.cont_range(self.data2plot,
                                   0, # offset
                                   ncont=31)
        a.contourf(self.data.t,
                   self.data.w,
                   self.data2plot.transpose(),
                   levels,
                   cmap=cm.seismic)
        a.plot([self.data.t[0],self.data.t[-1]],
               [self.data.w[self.idx_kin],self.data.w[self.idx_kin]],
               'k-')
        a.plot([self.data.t[self.idx_sp],self.data.t[self.idx_sp]],
               [self.data.w[0],self.data.w[-1]],
               'k-')

        aa.plot(self.data.t,self.data2plot[:,self.idx_kin])
        a4.plot(self.data.w,self.data2plot[self.idx_sp,:])
        a.set_xlim(self.data.t[0], self.data.t[-1])
        aa.set_xlim(self.data.t[0], self.data.t[-1])

        a.set_xlabel('$\mathregular{t\,[ps]}$',fontsize=12 )
        a.set_ylabel('$\mathregular{probe\,[cm^{-1}]}$',fontsize=12 )
        a.tick_params(axis='both', which='major', labelsize=10, pad=3)

        aa.set_xlabel('$\mathregular{t\,[ps]}$',fontsize=12 )
        aa.set_ylabel('$\mathregular{\Delta{A}\,[mOD]}$',fontsize=12 )
        aa.tick_params(axis='both', which='major', labelsize=10, pad=3)
        self.refreshFigure()

        

    def create_fit_fields_DAS(self, fields, inframe):
        entries = []
        row = 0
        for field in fields:
            lab = tk.Label(self, width=5, text=field)
            ent = tk.Entry(self,width=5)
            ent.insert(row, '1')
            
            lab.grid(row = row, column=0, in_=inframe)
            ent.grid(row = row, column=1, in_=inframe)

            entries.append((field, ent))
            row += 1
        tk.Button(self,
                  text='Fit',
                  command=(lambda e=entries: self.das(e))).grid(row=2,
                                                                column=0,
                                                                in_=inframe)

        tk.Checkbutton(self,
                    text='fit constant',
                    variable = self.fit_constant).grid(row=2,
                                                       column=1,
                                                       in_=inframe)
        return entries
        
