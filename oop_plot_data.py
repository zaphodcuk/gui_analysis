### Module with classes for dealing with data ###
#################################################

import os
import re
import math
import numpy as np
import pylab as plt
import time
import opusFC
import scipy.interpolate as interp
import scipy.optimize as optim
import pdb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog

import matplotlib
import matplotlib.cm as cm
import pathlib as pal

from sklearn.preprocessing import StandardScaler

################
### settings ###
################

###############################
### define useful constants ###
###############################

w_label = '\omega\,[cm^{-1}]'
w1_label = 'pump\,[cm^{-1}]'
w3_label = 'probe\,[cm^{-1}]'
abs_label = 'A\,[mOD]'
dabs_label = '\Delta{A}\,[mOD]'
t_label = 't\,[ps]'

##############################
##### Functions for FTIR #####
##############################
def extract_opus(files=None):
    if files == None:
        # open files dialogue
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(parent=root, title='Choose files')
        root.destroy()
        listf = []
        # find last instance of / and split path to fnames into listf
        for i in file_path:
            idx = i.rfind('/')
            listf.append(i[idx+1:])
        os.chdir(file_path[0][:idx]) # change directory to the one which contains files
        
    else:
        listf=files

    print(os.getcwd(), listf[0], listf[-1])
    for i in listf:
        dbs = opusFC.listContents(i)
        dat_abs = opusFC.getOpusData(i, dbs[0])
        dat_ref = opusFC.getOpusData(i, dbs[1])
        dat_sig = opusFC.getOpusData(i, dbs[2])
        dat = np.zeros((len(dat_abs.x),4))
        dat[:,0], dat[:,1], dat[:,2],dat[:,3] = dat_abs.x, dat_abs.y, dat_ref.y, dat_sig.y
        m=i.rfind(r'.')
        # distinguish repeated and single scan measurements.
        if (len(i[m:])>3): # four digit number -> repeated
            np.savetxt(str(i[:m]) +
                       '_' + str('{:03d}'.format(int(i[m+1:])))
                       + '.dat',dat)
        else:
            np.savetxt(str(i[:m]) +
                       '_' + str('{:02d}'.format(int(i[m+1:]))) +
                       '.dat',dat)


def interp_data_points(points,xdata,ydata):
    points2interp = np.zeros((len(points),2)) # array to fill with closest xy points
    j=0
    index = []
    for i in points:
        # closest points to the table
        index.append(np.argmin(abs(xdata-i)))
        # fill the array
        points2interp[j,0], points2interp[j,1] = xdata[index[j]], ydata[index[j]]
        j += 1

    ibeg = min(index)
    iend = max(index)
    new_y = interp.interp1d(points2interp[:,0], 
                            points2interp[:,1], 
                            kind='cubic')(xdata[ibeg:iend])

    # only interpolation, so resulting data are shorter.
    dataSubted = np.zeros((new_y.shape[0],2))
    # subtracting the interpol from the original
    dataSubted[:,0], dataSubted[:,1] = xdata[ibeg:iend], ydata[ibeg:iend]-new_y
        
    return dataSubted


def plot_init(list_of_files, ib = None, ie=None, list_idx = None, labels = None):
    if list_idx == None:
        lsit_idx = []
    if labels == None:
        labels = []
    # making a list of indeces to be plotted from the whole data set.
    if ib == None and ie == None and len(list_idx) == 0: # I plot everything
        ib = 0
        ie = len(list_of_files)
        list_idx = range(ib,ie,1)
    elif ib == None and ie == None and len(list_idx) != 0: # User defined sublist of the data
        pass
    elif ib != None and ie != None and len(list_idx) == 0: # some consecutive data between ib and ie
        list_idx = range(ib,ie,1)
    else:
        print('Wrong combination of inputs.')
        os.sys.exit()

    # matching the labels with the data.
    if labels != []:
        if len(labels) == len(list_of_files):
            pass
        else:
            if len(list_idx) != len(labels):
                print('labels have wrong lenght, I used default: filenames')
            else: pass

            lab = [0]*len(list_of_files)
            for i in range(len(list_idx)):
                lab[list_idx[i]] = labels[i]
            labels = lab
 
    else:
        for i in list_of_files:
            labels.append(i[-7:-4])

    return list_idx, labels

############
### Misc ###
############
def yes_or_no(question):
#    answer = input(str(question) + ' y/n').lower().strip()
    while "the answer is invalid":
        answer = input(str(question) + ' (y/n): ').lower().strip()
        if answer == 'y':
            return True
        elif answer == 'n':
            return False

def log_scale(log,t,t0):
    if log ==1:
        plt.xscale('log')
        if t0 ==0:
            plt.xlim(1e-1, t[-1])
        else:
            pass
    else:
        pass

def save_plot(name, save):
    if save == 1:
    # iterating until i find filename which does not exist yet.
        c = 0
        name2 = name
        while os.path.isfile(os.getcwd() + '/' +  str(name2) + '.png'):
            name2 = name + str('_{:02d}'.format(c))
            c += 1
        if c != 0:
            name = name2

        plt.savefig(str(name) + '.png', format="png")
        print('Plot saved as: ', str(name) + '.png')
    else:
        print('No plot saved')


##################
### Fitting PP ###
##################
def func(x,p,nexp):
    x=np.array(x)
    if nexp == 3:
        return (p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3]) + 
                p[4]*np.exp(-x/p[5]) + p[6])
    elif nexp == 2:
        return (p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3]) + 
                p[4])
    elif nexp == 1:
        return (p[0]*np.exp(-x/p[1]) + p[2])
    else:
        print("I can fit only 1,2 or 3 exponentials currently.")
        sys.exit()



def within_bounds(p,bounds):
    if len(p) == bounds.shape[0]:
        l = []
        for i in range(len(p)):
            l.append (str(bounds[i,0] <= p[i] <= bounds[i,1]))
        if any("False" in s for s in l):
            return 0
        else:
            return 1
    else:
        print("parametres and bounds length does not match")
        exit()



def residuals(p,y,x,nexp,bounds):
    x=np.array(x)
    cerr = np.dtype('Float64')
    if nexp == 3:
        if within_bounds(p,bounds):
            a1,t1,a2,t2,a3,t3,c = p
            cerr = y - (a1*np.exp(-x/t1) + a2*np.exp(-x/t2) + 
                        a3*np.exp(-x/t3) + c)
            return (cerr)
        else:
            return (1e12)
        
    elif  nexp == 2:
        if within_bounds(p,bounds):
            a1,t1,a2,t2,c = p
            cerr = y - (a1*np.exp(-x/t1) + a2*np.exp(-x/t2) + c)
            return (cerr)
        else:
            return (1e12)   
    elif nexp == 1:
        if within_bounds(p,bounds):
            a1,t1,c = p
            cerr = y - (a1*np.exp(-x/t1) + c)
            return (cerr)
        else:
            return (1e12)

def def_bounds(no_exp):
    if no_exp == 3:
        p0 = [[2e-2,200,5e-3,30,1e-3,900,1e-3]]
        bounds = np.array([[-100,100],[0.1,10000],
                           [-100,100],[0.1,10000],
                           [-100,100],[0.1,10000],
                           [-10,10]])
    elif no_exp == 2:
        p0 = [[5,2,3,0.5,1e-2]]
        bounds = np.array([[-100,100],[0.1,100],
                           [-100,100],[0.1,100],
                           [-10,10]])
    elif no_exp == 1:
        p0 = [[5,2,1e-2]]
        bounds = np.array([[-100,100],[0.1,100],
                           [-10,10]])

    return {'p0':p0, 'b':bounds}

def multiple_p0(no_exp):
    if no_exp == 3:
        p0 = [[5,0.5, 2,2, 1,20, 1e-2], # for ESA
              [0.5,0.5, 0.2,2, 0.1,20, 1e-2],
              [-5,0.5, -2,2, -1,20, -1e-2],
              [-0.5,0.5, -0.2,2, -0.1,20, -1e-2],
              [-0.5,2, 0.3,5, 0.1,20, 1e-2],
              [0.5,2, -0.3,5, -0.1,20, -1e-2]]

    elif no_exp == 2:
        p0 = [[5,2, 3,20, 1e-2],
              [0.5,3, 0.2,20, 1e-2],
              [-5,2, -3,20, 1e-2],
              [-0.5,3, -0.2,20, 1e-2],
              [-0.5,2, 0.3,5, 1e-2],
              [0.5,2, -0.3,5, 1e-2]]

    elif no_exp == 1:
        p0 = [[5,2, 1e-2],  # ESA strong
              [0.5,3, 1e-2], # ESA weak
              [-5,2, 1e-2], # GSB strong
              [-0.5,3, 1e-2]]# GSB weak
    return p0
              
###################
### Plotting PP ###
###################
def plot_snippet1D_init(xlabel,ylabel,plot_wid=0):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('$\mathregular{' + xlabel + '}$',fontsize=18 )
    ax.set_ylabel('$\mathregular{' + ylabel + '}$',fontsize=18 )
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    if plot_wid == 1:
        canvas = FigureCanvasTkAgg(fig)
    else:
        pass
    return ax

def plot_snippet1D_end(axis, save=0, name = 'plot', plot_wid = 0):
    # for many labels I shift the legend outside
    if len(axis.get_legend_handles_labels()[1]) < 7:
        plt.legend(loc=1, borderaxespad=0.,handlelength=1.2)
    else:
        plt.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.,handlelength=1.2)
#    print(axis.get_legend_handles_labels()[1])
    plt.tick_params(axis='both', which='major', labelsize=15, pad=6)
    plt.margins(0.01)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.92)
    save_plot(name, save)
    if plot_wid == 0:
        plt.show()
    elif plot_wid ==1:
        canvas.get_tk_widget().pack()
        canvas.draw()
    else:
        print('invalid option for canvas')


def plot_snippet2D_init(xlabel,ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\mathregular{' + xlabel + '}$',fontsize=18 )
    ax.set_ylabel('$\mathregular{' + ylabel + '}$',fontsize=18 )

def plot_snippet2D_end(axis, save=0, name = 'plot2d'):
    plt.tick_params(axis='both', which='major', labelsize=15, pad=6)
    plt.tight_layout()
#    plt.gcf().subplots_adjust(left=0.15, bottom=0.11)
    save_plot(name, save)
    plt.show()


def cont_range(data, off, ncont, mode='auto', const=1):
    ncont = abs(round(ncont)) # in the case of negative and non-integer numbers
    if mode == "auto":
        mx=np.amax(np.subtract(data,off))
        mn=np.amin(np.subtract(data,off))
        # positive signal larger
        if abs(mx) >= abs(mn):
            if mx > 0: # mx is negative
                lev = np.linspace(-mx,mx,ncont);
            else: # mx is positive
                lev = np.linspace(mx,-mx,ncont);
        else: # abs negative signal is larger
            if mn < 0: # min negative
                lev = np.linspace(mn,-mn,ncont);
            else: # min positive
                lev = np.linspace(-mn,mn,ncont);
    elif mode == "man":
        lev = np.linspace(-const,const,ncont)
    else:
        print("Error: unknown \"mode\" given, default levels set between -1 and 1")
        lev = np.linspace(-1,1, ncont)

    return lev


def plot1Dfrom2D(data, value_list, ax_to_cut, ax0, ax1, const=0, offset=0, x0=0):
    for i in range(len(value_list)):
        cmap = cm.gist_heat((i)/len(value_list),1)
        if ax_to_cut == 0: ## cutting at rows, which means spectra
            idx_value = np.argmin(abs(ax0-const-value_list[i]))
            idx_x0 = np.argmin(abs(ax1-x0))
            plt.plot(ax1[idx_x0:],
                     np.subtract(data,offset)[idx_value,idx_x0:],
                     label = round(ax0[idx_value]-const,2),
                     color = cmap,
                     linewidth=2)

        elif ax_to_cut == 1: ## cutting at columns, which means kinetics
            idx_value = np.argmin(abs(ax1-const-value_list[i]))
            idx_x1 = np.argmin(abs(ax0-x0))
            plt.plot(ax0[idx_x1:],
                     np.subtract(data,offset)[idx_x1:, idx_value],
                     label = round(ax1[idx_value]-const,0),
                     color = cmap,
                     linewidth = 2)
        else:
            print('Invalid value of ax_to_cut.')
            sys.exit()


def plot_pp(data,t,w,title, off = 0, ncont = 31, t0_plot=0):
    levels = cont_range(data, off, ncont)
    plt.xlim(t0_plot, t[-1])
    plt.contourf(t,w,np.subtract(data,off).transpose(),levels,cmap=cm.seismic)
    if yes_or_no('Save the plot?'):
        plt.savefig(title + '.png')
    else:
        pass



def plot2Dfrom3D(matrix,x,y,offset,levels,idx,name,save, ticks = [-0.4, 0, 0.4]):
    X, Y = np.meshgrid(x, y)
    if isinstance(name, str):
        fig.suptitle(name, fontsize=14, fontweight='bold')
    
    plt.contourf(X,Y,
                 np.transpose(matrix[:,:,idx])-offset,
                 levels,
                 cmap=cm.seismic)

    cbar = plt.colorbar(shrink=0.98, pad=0.03)
#    cbar.ax.set_yticklabels(ticks,fontsize=20, fontweight = 'normal')
#    cbar.set_ticks(ticks)
    if save == 1:
        plt.savefig(str(name) + '.png', format="png")
        print('Plot saved as: ', str(name) + '.png')
    else:
        print('No plot saved')

###################
### 2D analysis ###
###################
def step(x):
    return 1 * (x > 0)


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


###############
### Classes ###
###############

class Data():
    # I want to know if it is data file, extract the delimiter, header and data
    # data in the shape of list of arrays
    def __init__(self, files=None , header = None, raw_data=None, dim=None):
        if files == None:
            # load files dialogue
            root = tk.Tk()
            root.withdraw()
            # list of files which includes whole path.
            files_full = filedialog.askopenfilenames(parent=root, title='Choose files')
            root.destroy()

            idx = files_full[0].rfind('/') # finds last instance of /
            listf = []
            for i in files_full:
                listf.append(i[idx+1:])
            self.files = listf
            os.chdir(files_full[0][:idx]) # change of directory only when list is not inputed
        else:
            self.files = files


        ## Header assingments
        self.header = []
        self.raw_data = []
        self.dim = []
        for i in self.files:
            iffloat = False
            f = open(i)
            h=[]
            while iffloat == False:
                txt = f.readline()
                txt2 = re.split(r'[ ,|;"]+', txt)
                for j in range(len(txt2)):
                    try:
                        a = float(txt2[j])
                    except ValueError:
                        h.append(txt2[j])
                    else:
                        iffloat = True
            f.close()
            self.header.append( " ".join(h)  )


        ## data assingment
        for i in range(len(self.files)):
            hlen = len(self.header[i].split('\n'))
            dd = np.loadtxt(self.files[i],skiprows=hlen)
            self.raw_data.append( dd )
            ## assing dim of each of the data here.
            self.dim.append( dd.shape )


    
    def get_name(self):
        print("File name is:", self.files)
        return self.files


    def get_header(self):
        print(self.header)
        return self.header


    def print_data(self):
        print("The data of the files are:", self.raw_data)

    def get_data(self):
        if self.raw_data == None:
            print("no data identified yet:", self.raw_data)
        else:
            return self.raw_data

    def get_dim(self):
        print(self.dim)
        return self.dim



    # I want general plotter, specific plotting will be by subclasses.
    def plot1D(self, xlabel = 'x', ylabel = 'y',ymulti=1, save = 0, name = 'plot'):
        c = 0
        for i in self.files:
            ax = plot_snippet1D_init(xlabel,ylabel)
            if len(self.dim[c]) > 3:
                print("I cannot deal with more than 3dim data")
                exit()
            # data contains more arrays, dim is a n array of arrays dimensions
            elif len(self.dim[c]) == 3:
                for j in range(self.raw_data[c].shape[2]):
                    print(value_list)
                    cmap = cm.gist_heat((i) / self.raw_data[c].shape[2])

                    if (self.dim[c][1] == 2):
                        # only single 1D data, but multiple arrays
                        print("plotting single 1D data")
                        plt.plot(self.raw_data[c][:,0,j], self.raw_data[c][:,1,j]*ymulti, color=cmap)
                    else:
                        # i will plot it as 2D
                        print("plotting as 2D data")
                        plt.plot(self.raw_data[c][:,0,j], self.raw_data[c][:,1:,j]*ymulti, color=cmap)
                    plot_snippet1D_end(ax, save, name)

            else:
                # data contain only one array
                if (self.dim[c][1] == 2):
                    # only single 1D data
                    print("plotting single 1D data")
                    plt.plot(self.raw_data[c][:,0], self.raw_data[:,1]*ymulti)
                else:
                    # i will plot it as 1D
                    print("plotting as 2D data")
                    plt.plot(self.raw_data[c][:,0], self.raw_data[c][:,1:]*ymulti)
            c += 1
            plot_snippet1D_end(ax, save, name)
        


        

 
class FTIR(Data): # new format has always 4 columns, energy, abs, ref, sig
    def __init__(self, files=None,header = None, raw_data=None, data = None, dim=None, gl_ref = None, corr_table = None):
        super(FTIR, self).__init__(files,header = None, raw_data=None, dim=None)
        list_sin = []
        list_rep = []
        for i in self.files:
            if i[-7] == '_':  # this is for single scans
                list_sin.append(i)
            elif i[-8] == '_': # this is for repeated scans
                list_rep.append(i)
            else:
                print('wrong format of the file name')
                exit()
        self.files = list_sin + list_rep  # full list

        self.gl_ref = None
        self.corr_table = []  # this is the last correction performed on the data.
        self.data = [] # this is the current data to plot which are just two columns, x and y.
        for i in range(len(self.files)): # calculating absorption from the raw_data
            d = np.zeros(( self.raw_data[0].shape[0],2 ))
            d[:,0] = self.raw_data[i][:,0]
            d[:,1] = np.log10(self.raw_data[i][:,2]/self.raw_data[i][:,3])
            self.data.append(d)

#####################

    ## set by index of the data
    def glob_ref(self,idx):
        if self.gl_ref == None:
            self.gl_ref = idx # after this I have to change the "data" if it was corrected
            if len(self.corr_table) > 0: # correction done before I have to rewrite 'data'
                self.bcg_corr(table = self.corr_table, change = 1)
                print('Global ref set to', idx, 'and bcg_corr called to update "data"')
            else:
                print('Global ref set to', idx)
        else:
            if yes_or_no('Gl_ref already exist, wanna change it?'):
                self.ch_glob_ref(idx)
                if len(self.corr_table) > 0: # correction done before I have to rewrite 'data'
                    self.bcg_corr(table = self.corr_table, change = 1)
                    print('Global ref set to', idx, 'and bcg_corr called to update "data"')
                else:
                    print('Global ref set to', idx)
            else:
                print('Gl_ref not changed, use ch_gl_ref if you change your mind.')


    def rem_glob_ref(self):
        if self.gl_ref != None: # remove gl_ref
            self.gl_ref = None
            if len(self.corr_table) > 0: # I have to call correction again
                self.bcg_corr(table = self.corr_table, change =1)
                print('Global ref disabled, bcg_corr called to update "data".')
            else:
                print('Global ref disabled')


        else: # no change, no call of the corr again
            print('Gl_ref does not exist, so nothing removed')


    def ch_glob_ref(self,idx):
        if self.gl_ref != None:
            if idx == self.gl_ref:
                print('Same global ref as before, I do nothing.')
            else:
                self.gl_ref = idx
                if len(self.corr_table) > 0: # I have to call correction again
                    self.bcg_corr(table = self.corr_table, change =1)
                    print('Global ref changed to', idx, 'and bcg_corr again.')
                else:
                    print('Global ref changed to', idx)
                    
        else:
            print('Gl_ref does not exist')
                        

    def print_current_data_status(self):
        # print if global reference is active and 
        if (self.gl_ref != None and len(self.corr_table) > 0):
            print('Corr data with global ref')
        elif (self.gl_ref != None and len(self.corr_table) == 0):
            print('UNcorr data with global ref')
        elif (self.gl_ref == None and len(self.corr_table) > 0):
            print('Corr data withOUT global ref')
        elif (self.gl_ref == None and len(self.corr_table) == 0):
            print('UNcorr data withOUT global ref')
        else:
            print('Unrecognized option in "print_current_data_status()" ')



    # how to combine it with a global reference option
    # corr data come only with two columns
    def bcg_corr(self, table = [1000,1200,1930,2230,2500,2700,3900], change = 0):
        if len(self.corr_table) > 0: # correction exist
            if change == 1:
                if self.gl_ref != None: # gl_ref exist I take it into calculation
                    self.data = []
                    for i in range(len(self.files)): # corr of all the files.
                        self.data.append(
                            interp_data_points(
                                table,
                                self.raw_data[i][:,0],
                                np.log10(self.raw_data[self.gl_ref][:,2]/self.raw_data[i][:,3]) ))
                    print('corr rewriten, gl_ref used.')

                else: # gl_ref does not exist
                    self.data = []
                    for i in range(len(self.files)): # corr of all the files.
                        self.data.append(
                            interp_data_points(
                                table,
                                self.raw_data[i][:,0],
                                np.log10(self.raw_data[i][:,2]/self.raw_data[i][:,3]) ))
                    print('corr rewritten, gl_ref NOT used.')

            else: # correction exist and user does not want to rewrite it
                print('correction not changed, use "change = 1" option if you change your mind')

        else: # correction does not exist
            if self.gl_ref != None: # gl_ref exist I take it into calculation
                self.data = []
                for i in range(len(self.files)): # corr of all the files.
                    self.data.append(
                        interp_data_points(
                            table,
                            self.raw_data[i][:,0],
                            np.log10(self.raw_data[ self.gl_ref ][:,2]/self.raw_data[i][:,3]) ))
                print('new corr, gl_ref used')

            else: # gl_ref does not exist
                self.data = []
                for i in range(len(self.files)): # corr of all the files.
                    self.data.append(
                        interp_data_points(
                            table,
                            self.raw_data[i][:,0],
                            np.log10(self.raw_data[i][:,2]/self.raw_data[i][:,3]) ))
                print('new corr, gl_ref NOT used')


        # update self.corr_table for the table I just performed
        self.corr_table = table

    def rem_bcg_corr(self):
        self.data = []
        self.corr_table = []
        for i in range(len(self.files)): # calculating absorption from the raw_data
            d = np.zeros(( self.raw_data[0].shape[0],2 ))
            d[:,0] = self.raw_data[i][:,0]
            d[:,1] = np.log10(self.raw_data[i][:,2]/self.raw_data[i][:,3])
            self.data.append(d)
        if self.gl_ref != None:
            self.glob_ref(self.gl_ref)
        else:
            pass




    # I always plot from ref and sig, I know each file is just a 2d array
    def plot(self, ib = None, ie=None, list_idx = None, labels = None, ymulti = 1, save=0, name = 'plot'):
        # taking care of labels and if I plot interval, or some separate plots, corr etc.
        if list_idx == None:
            list_idx = []
        if labels == None:
            labels = []
        
        list_idx,labls = plot_init(self.files, ib, ie, list_idx, labels)
        
        #plotting
        llen = len(list_idx)
        ax = plot_snippet1D_init(w_label,abs_label)
        for i in range(llen):
            cmap = cm.gist_heat((i)/llen,1)
            plt.plot( self.data[list_idx[i]][:,0],
                      self.data[list_idx[i]][:,1]*ymulti,
                      color = cmap,
                      label = labls[list_idx[i]] )
            print(labls[list_idx[i]] )

        plot_snippet1D_end(ax, save, name)
        self.print_current_data_status()



    def pca(self, subdata = None):
        subdata = subdata if subdata is not None else self.data
        dmatrix = [np.log10(subdata[0][:,2]/subdata[0][:,3])]
        for i in range(len(subdata)-1):
            dmatrix = np.vstack((dmatrix,np.log10(subdata[i+1][:,2]/subdata[i+1][:,3])))
        dmatrix = dmatrix.transpose()
        print(dmatrix.shape)
        X_std = StandardScaler().fit_transform(dmatrix)
        mean_vec = np.mean(X_std, axis=0)
        cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
        #print('Covariance matrix \n%s' %cov_mat)
        cov_mat = np.cov(X_std.T)

        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        #print('Eigenvectors \n%s' %eig_vecs)
        #print('\nEigenvalues \n%s' %eig_vals)
        u,s,v = np.linalg.svd(X_std.T)
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        print('Everything ok!')
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        print('Eigenvalues in descending order:')
        #for i in eig_pairs:
        #    print(i[0])
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        matrix_w = np.hstack((eig_pairs[0][1].reshape(dmatrix.shape[1],1), 
                              eig_pairs[1][1].reshape(dmatrix.shape[1],1),
                              eig_pairs[2][1].reshape(dmatrix.shape[1],1),
                              eig_pairs[3][1].reshape(dmatrix.shape[1],1),
                              eig_pairs[4][1].reshape(dmatrix.shape[1],1),
                              eig_pairs[5][1].reshape(dmatrix.shape[1],1)))

        #print('Matrix W:\n', matrix_w)
        Y = X_std.dot(matrix_w)
        print(matrix_w.shape)
        plt.ion()
        for i in range(Y.shape[1]):
            plt.plot(subdata[0][:,0], Y[:,i])
        plt.show()
        plt.ioff()
        plt.savefig('pca.png', format='png')
        plt.clf()



class CV(Data):
    def __init__(self):
        pass

    def plot(self):
        pass



######################
######################
class PP(Data):
    def __init__(self, dname = None, format_data = None, data = None, avdata = None, single_scan_data = None, Nscans = None,t = None, w = None, t0 = None, Nsm = None, sm = None, dim = None, offset = None):
        # TODO make single scan_data_work
        # either input directory or choose in the dialogue
        if dname == None:
            root = tk.Tk()
            root.withdraw()
            directory = filedialog.askdirectory(parent=root, title='Choose a directory')
            root.destroy()
        else:
            directory = dname
        print(directory)
        os.chdir(directory)

        # initialization
        self.dname = directory[directory.rfind('/')+1:]
        self.data = []
        self.sm = []
        self.format_data = format_data
        
        # checking if all the temp files are there.
        if (format_data == 'lab4'):
            foolist = [f for f in os.listdir(str(pal.Path.cwd() / 'temp')) if not re.search('~',f)]
        elif (format_data == 'lab1'):
            foolist = [f for f in os.listdir(os.getcwd() + 'temp') if not re.search('~',f)]
        else:
            pass

        if (len(foolist) % 3) == 0:
            pass
        else:
            print('Warning: Number of files in temp is not dividable by 3')

        print(len(foolist))
        self.Nscans = (int(len(foolist)))
        self.t = np.loadtxt(self.dname+'_delays.csv')/1000
        self.w = np.loadtxt(self.dname+'_wavenumbers.csv')
 
        # read slow modulation file
        f = open(self.dname+'_slowModulation.csv')
        self.Nsm = int(f.readline()) # this is the number of modulation points, first line of the file
        d={}
        while True:
            l = f.readline()
            if not l: break
            idx = l.find(':') # find a : in the line
            listfoo = l[idx+3:-2].split(',')
            for i in range(len(listfoo)):
                listfoo[i] = float(listfoo[i])
            d[l[:idx]] = listfoo
        self.sm = d
        f.close()

        # ignoring 'sp' and 'du'
        lfoo = sorted(f for f in os.listdir(os.getcwd()) if re.search(self.dname + '_signal_sp0_',f))
        if len(lfoo) == self.Nsm:
            pass
        else:
            print('Warning: N signal files does not match "sm" value.')
            print(lfoo, self.Nsm)
        dfoo = np.loadtxt(self.dname+'_signal_sp0_sm0_du0.csv', delimiter=',')
        dat = np.zeros((dfoo.shape[0], dfoo.shape[1],len(lfoo)))
        for i in range(len(lfoo)):
            dat[:,:,i] = np.loadtxt(self.dname+'_signal_sp0_sm' + str(i) + '_du0.csv', delimiter=',')
        self.data = dat
        self.avdata = np.sum(self.data, axis=2)
        self.offset = np.zeros(len(self.avdata[0,:]))
        ## TODO, load single scan data
            
    def def_offset(self,data, method = 'auto'):
        # options are 'auto' (last population time point)
        # 'negative' (subtract first measured point, typically -20ps)
        # 'const' (subtract average of the last scan)
        # works only for 2d data, not 3d set of pp
        if method == 'auto':
            self.offset = data[-1,:]
        elif method == 'negative':
            self.offset = data[0,:]
        elif method == 'const':
            self.offset = np.ones( len(data[-1,:]) )*np.sum(data[-1,:])/len(data[-1,:])
        else:
            print('Unknown method for defining PP offset.')
#        return offset
        

    def plot(self, data, ncont=31, ymulti=1, save=0, name='plot', log=0, t0=0, canvas = 0):
        print('Shape of the plotted data:', data.shape)
        ax = plot_snippet1D_init(t_label,dabs_label,plot_wid=canvas)
        if len(data.shape) == 2:
            plot_pp(data, self.t, self.w, self.dname, off=self.offset, ncont=ncont, t0_plot=t0)
        else: # not sure if I still allow for this (maybe after implementing the single scan plots)
            for i in range(data.shape[2]):
                plot_pp(data[:,:,i]*ymulti, self.t, self.w, self.dname, off=self.offset)
        
        log_scale(log, self.t, t0)
        plot_snippet1D_end(ax, save, name, plot_wid = canvas)




    def plot_spectra(self, data, popTimes = [0.3,0.5,1], 
                     offset = 0, const = 0, ymulti = 1, 
                     save=0, name = 'plot'):
        ax = plot_snippet1D_init(w1_label, dabs_label)
        plot1Dfrom2D(data*ymulti,popTimes, 0, self.t, self.w, const, offset)
        plot_snippet1D_end(ax, save, name)



    def plot_kin(self,data, wnums = [2050], offset = 0,
                 const = 0, ymulti = 1, save=0, name = 'plot',log=0,t0=0): # of the averaged data

        ax = plot_snippet1D_init(t_label, dabs_label)
        plot1Dfrom2D(data*ymulti,
                     wnums,
                     ax_to_cut=1, 
                     ax0=self.t, ax1=self.w, 
                     const=const,
                     offset=offset,
                     x0=t0)
        log_scale(log,self.t,t0)
        plot_snippet1D_end(ax, save, name)



    def fit_kin(self, wavenums,nexp, tinit,xlabel, ylabel, ymulti=1, save=0, name='plot',log=0,t0=0,offset =0):
        # TODO implement fits with diff imitial params p0(GSB/ESA etc.) and pick the best one
        # this is messed up right now.
        fit=[]
        bounds = def_bounds(nexp)['b']
        p0 = multiple_p0(nexp)
        iinit = np.argmin(abs(self.t-tinit))
        ax = plot_snippet1D_init(xlabel,ylabel)
        for i in range(len(wavenums)):
            ff = []
            res = []
            iw = np.argmin(abs(self.w-wavenums[i]))
            for j in range(len(p0)):
                ff.append(optim.leastsq(residuals,p0[j],
                                      args = (np.subtract(self.avdata,offset)[iinit:,iw]*ymulti,
                                              self.t[iinit:],nexp,bounds),
                                      maxfev=1000),)
                res.append(np.sum((np.subtract(self.avdata,offset)[iinit:,iw]*ymulti- func(self.t[iinit:]-t0,ff[j][0],nexp))**2))
            print(res)

            
            fit.append (optim.leastsq(residuals,p0[j],
                                      args = (np.subtract(self.avdata,offset)[iinit:,iw]*ymulti,
                                              self.t[iinit:],nexp,bounds),
                                      maxfev=1000),)
            plt.plot(self.t[iinit:]-t0,
                     np.subtract(self.avdata,offset)[iinit:,iw]*ymulti,linewidth=2,
                     label="{0:.2f}".format(fit[i][0][1]) + '@' + "{0:.0f}".format(self.w[iw]))
            plt.plot(self.t[iinit:]-t0,
                     func(self.t[iinit:]-t0,fit[i][0],nexp),
                     'k')
        print(round(fit[0][0][1],2))
        log_scale(log,self.t, t0)
        plot_snippet1D_end(ax, save, name)

    # TODO for sure.
    def remove_mean(self):
        self.data_wto_mean = np.zeros(self.avdata.shape)
        for i in range(self.data_wto_mean.shape[0]):
            self.data_wto_mean[i,:] = self.avdata[i,:]-np.mean(self.avdata[i,:])


    
    def fit_kin_glob():
        pass

    def subtract_one_from_all(self, idx):
        pass


######################
######################
class MD(Data):
    def __init__(self, dname=None, format_data=None, pref = None, bins = None, data = None, avdata = None, single_scan_data = None, Nstates = None,t = None, w = None, t0 = None, Nsm = None, sm = None, dim = None):
        #### constants ####
        #self.filter_type     = 'Mean'
        #self.filter_points   = 10
        self.zeropad_npoints = 10
        self.phase_points = 50
        #self.apodize_gaussian= 0
        #self.apodize_Gcoeff  = 1000
        #self.probe_fitorder  = 1
        #self.phase_fitmethod = 'Shift bins' # 'Shift wavenumbers' or 'Shift bins'
        #self.phase_wavenum   = 100
        self.phase_shiftbins = 10
        # Physical constants
        self.HeNe            = 2.11079          # HeNe period (fs)
        self.c_0             = 2.99792458e-5    # Speed of light in cm/fs
        self.Npixels = 32

        metaf = []
        # in cwd find all metafiles and make a list of PP data files.
        if dname == None:
            root = tk.Tk()
            root.withdraw()
            ddir = filedialog.askdirectory(parent=root, title='Choose a directory')
            root.destroy()
        else:
            ddir = dname
        os.chdir(ddir)

        metaf = [f for f in os.listdir(os.getcwd()) if re.search('_meta.txt',f)
                 and not re.search('~',f)]
        if len(metaf) == 1:
            pass
        else:
            print('Warning: more than one metafile.')
        self.pref = metaf[0][:-9]
        print(self.pref)

        self.bins = np.loadtxt(self.pref + '_bins.csv', delimiter = ',')[:,1]             #bins in fs
        self.Nbins = len(self.bins)
        
        self.Nstates = int(np.loadtxt(self.pref + '_Ndatastates.csv'))   #N states -> no of matrices
        
        self.data = []
        self.sm = []
        self.t = np.loadtxt(self.pref+'_delays.csv')/1000           # time axis
        try:
            self.lt = len(self.t)
        except TypeError:
             self.lt = 1   
        self.w = np.loadtxt(self.pref+'_wavenumbers.csv')           # wavenumber axis
        f = open(self.pref+'_slowModulation.csv')
        self.Nsm = int(f.readline()) # this is the number of modulation points, first line of the file
        d={}
        while True:
            l = f.readline()
            if not l: break
            idx = l.find(':') # find a ':' in the line
            listfoo = l[idx+3:-2].split(',')
            for i in range(len(listfoo)):
                listfoo[i] = float(listfoo[i])
            d[l[:idx]] = listfoo
        self.sm = d
        f.close()
        
        # loading the real stuff
        # ignoring 'sp' and 'in'
        temp_interf = np.zeros(( self.Nbins, self.Nstates ))
#        temp_interfMCT = np.zeros(( self.Nbins, self.Nstates ))
        self.counts = np.zeros(( self.Nbins, self.Nsm, self.lt, self.Nstates ))
        self.interf = np.zeros(( self.Nbins,self.Nsm, self.lt ))
#        self.interfMCT = np.zeros(( self.Nbins, self.Nsm, self.lt ))
        self.probe = np.zeros(( self.Nbins,self.Npixels, self.Nsm, self.lt, self.Nstates ))
        self.ref = np.zeros(( self.Nbins,self.Npixels, self.Nsm, self.lt, self.Nstates ))
        self.signal = np.zeros(( self.Nbins,self.Npixels, self.Nsm, self.lt ))
        # splitting the for loop in order to check for empty bins
        self.zeros = []
        for i in range(self.Nsm):
            for j in range(self.lt):
                for k in range(self.Nstates):
                    self.counts[:,i,j,k] = np.loadtxt(self.pref + '_count_ds' + 
                                                      str(k) + '_sp0_sm' + str(i) + 
                                                      '_de' + str(j) + '_in0.csv')
                    for l in range(len(self.counts[:,i,j,k])):
                        if self.counts[:,i,j,k][l] == 0:
                            self.zeros.append(l)

        if len(self.zeros) == 0:
            pass
        else:
            print('2D corrupted, some counts are zero')
            print(self.zeros)
        # I want to divide everything by counts right away.
        for i in range(self.Nsm):
            for j in range(self.lt):
                for k in range(self.Nstates):
                    # loading files, pyro, , probe 
                    # I do not know the difference between the first and second column.
                    temp_interf[:,k] = np.loadtxt(self.pref +
                                                     '_interferogram_ds' + str(k) + 
                                                     '_sp0_sm' + str(i) + '_de' + str(j) + 
                                                     '_in0.csv', 
                                                     delimiter = ',')[:,0]/ self.counts[:,i,j,k]
                    
                    probe = np.loadtxt(self.pref + '_probe_ds' + str(k) + '_sp0_sm' + 
                                          str(i) + '_de' + str(j) + '_in0.csv', delimiter = ',')

                    ref = np.loadtxt(self.pref + '_reference_ds' + str(k) + '_sp0_sm' + 
                                        str(i) + '_de' + str(j) + '_in0.csv', delimiter = ',')

                    # dividing probe/ref by counts to make signal independent of MCT
                    probe = probe/ self.counts[:,i,j,k][:,None]
                    ref = ref/ self.counts[:,i,j,k][:,None]
                    
                    # divide by mean to make signal independent of MCT
                    self.probe[:,:,i,j,k] = probe / np.mean(probe,axis = 1)[:,None]
                    self.ref[:,:,i,j,k] = ref / np.mean(ref,axis=1)[:,None]
                    

                # chopper off
                if self.Nstates == 1:
                    self.interf[:,i,j] = temp_interf[:,None]
#                    self.interfMCT[:,i,j] = temp_interfMCT
                    #signal in mOD
                    self.signal[:,:,i,j] = -1000*np.log10(self.probe[:,:,i,j,k] / 
                                                          self.ref[:,:,i,j,k])

                # Chopper on, doing on-off for the signal, reducing dimension
                elif self.Nstates == 2:
                    # 0 is pump on, 1 is pump off
                    self.interf[:,i,j] = temp_interf[:,0] - temp_interf[:,1]
#                    self.interfMCT[:,i,j] = temp_interfMCT[:,0] - temp_interfMCT[:,1]
                    self.signal[:,:,i,j] = -1000*( np.log10(self.probe[:,:,i,j,0] / 
                                                            self.ref[:,:,i,j,0]) - 
                                                   np.log10(self.probe[:,:,i,j,1] / 
                                                            self.ref[:,:,i,j,1]) )
                else:
                    print('I cannot cope with this number of "Nstates".')

                    
        # average over slow modulation
        self.avinterf = np.sum(self.interf, axis=1)/self.Nsm
#        self.avinterfMCT = np.sum(self.interfMCT, axis=1)/self.Nsm
        self.avprobe = np.sum(self.probe, axis=2)/self.Nsm
        self.avref = np.sum(self.ref, axis=2)/self.Nsm
        self.avsignal = np.sum(self.signal, axis=2)/self.Nsm

        # resolution and pump axis
        self.resolution = 1/(self.Nbins*self.HeNe*self.c_0)
        self.ax_pump = np.arange(self.Nbins)*self.resolution
        # initialization
        self.fftAvInterf = np.zeros(( self.Nbins,self.lt ), dtype = np.complex64)
        self.maxAvInterf = []
        self.maxFFtAvInterf = []
        # remove mean from the data and interferogram
        for i in range(self.lt):
            # Ricardo has a -sign here, WHY?
            # subtract interf mean and get bin of the max
            self.avinterf[:,i] = -(self.avinterf[:,i] - np.mean(self.avinterf[:,i]) )
            self.maxAvInterf.append( np.argmax(self.avinterf[:,i]) )
            # pump spectrum and its max bin
            self.fftAvInterf[:,i] = abs(np.fft.fft(self.avinterf[:,i]))
            self.maxFFtAvInterf.append( 20+np.argmax(
                np.absolute(self.fftAvInterf[20:int(np.floor(len(self.fftAvInterf[:,i])/2)),i])))
            
            for j in range(self.Npixels):
                self.avsignal[:,j,i] = self.avsignal[:,j,i] - np.mean( self.avsignal[:,j,i] )
        
        print('maxAvInterf',self.maxAvInterf)
#        print(self.signal.shape, self.avsignal.shape)




    def apodize(self, method = 'cos2'): 
        apodize = np.zeros(self.avinterf.shape)
        box = np.zeros(self.avinterf.shape)
        self.apo_signal = np.zeros(self.avsignal.shape)
#        plt.plot( self.avsignal[:,18,0] )

        for j in range(self.lt):
            apodize[:,j] = np.power(np.cos(np.pi*(np.arange(self.Nbins)-self.binZero[j])/
                                           (2*(self.Nbins-self.binZero[j]))),2)
            box[:,j] = step(np.arange(self.Nbins) - self.binZero[j])#np.heaviside(self.bins - self.binZero[j],0.5)
        # apodize without box to get the phase later on
        self.avinterf = np.multiply(self.avinterf,apodize)
        for i in range(self.Npixels):
            self.avsignal[:,i,:] = np.multiply( apodize,np.multiply(box,self.avsignal[:,i,:]))
        
#        plt.plot( self.avsignal[:,18,0] )
#        plt.plot(apodize)
#        plt.title('apodization')
#        plt.show()
        
        
        
    def padding(self, interf, data, pad = 0):
        closestpow = nextpow2(self.Nbins)
        NFTpoints = closestpow * 2**pad
        
        self.resolution = 1/(NFTpoints * self.HeNe * self.c_0)
        self.ax_pump = np.arange(NFTpoints) * self.resolution
        
        self.avinterf = np.zeros((NFTpoints, self.lt))
        self.avsignal = np.zeros((NFTpoints, self.Npixels, self.lt))

        self.avinterf[:self.Nbins, :] = interf
        self.avsignal[:self.Nbins,:,:] = data

        self.Nbins = NFTpoints

        

    def plot2D(self, data, pump, offset = 0, ncont = 41, save = 0, name = 'plot2d'):
        # find indeces of the pump axis matching the probe
        ib = np.argmin( abs(pump - self.w[0]) )
        ie = np.argmin( abs(pump - self.w[-1]) ) +1
        print(ib,ie)
        
        self.plot_data = np.zeros( data.shape )
        for j in range(self.lt):

            levels = cont_range(np.real(data[ib:ie,:,j]),offset,ncont, mode = 'auto')

            ax = plot_snippet2D_init(w1_label, w3_label)
            plot2Dfrom3D(np.real(data[ib:ie,:,:]),
                     pump[ib:ie],
                     np.arange(self.Npixels),
                     offset = offset,
                     levels = levels, 
                     idx = j, 
                     name = self.t[j],
                     save = 1 )

            plot_snippet2D_end(ax, save=0, name = 'plot2d')


    def pre_phase(self):
        self.binZero = []
        for j in range(self.lt):
            tbin = []
            tdiff = []
            for i in range(200):
                tbin.append(i+self.maxAvInterf[j] -100)
                tempPhase = np.unwrap(np.angle(np.fft.fft(np.roll(self.avinterf[:,j],-tbin[i]))))
                tdiff.append(tempPhase[self.maxFFtAvInterf[j] + self.phase_shiftbins] 
                             -tempPhase[self.maxFFtAvInterf[j] - self.phase_shiftbins])

            # min difference between two phases
            self.binZero.append( tbin[np.argmin(np.absolute(tdiff))])

        print('binzero, prephase',self.binZero)



    def phase(self, data, interf,pump_corr = 1):
        # apply to zero padded (ZP) objects.
        print('interf.shape', interf.shape)
        inter = np.zeros( interf.shape )
        fft_inter = np.zeros( interf.shape, dtype = np.complex64  )
        pump = np.zeros( interf.shape )

        self.fft_sig = np.zeros( data.shape, dtype = np.complex64 ) # FT of of the signal
        
        self.phase = np.zeros( interf.shape ) # phase correction
        self.pump_max_bin = [] # bin of the pump maximum.
        
        shift_fitphase = np.around(self.phase_points/self.resolution)
        fitted_phase = np.zeros( interf.shape )
        phasing_term = np.zeros( interf.shape, dtype = np.complex64 )
        idx_start = []
        idx_end = []
        phase_coeff = []
        print('shift_fitphase', shift_fitphase)
        print('resolution', self.resolution)
        for i in range(self.lt):
            # move the negative time to the end, NOT sure it should not be just cut out
            inter[:,i] = np.roll(interf[:,i],-int(self.binZero[i]))
            fft_inter[:,i] =np.fft.fft(inter[:,i])
            pump[:,i] = abs(fft_inter[:,i])  # pump spectrum
            self.pump_max_bin.append( 100 + np.argmax( pump[:,i][100:int(np.floor(len(pump[:,i])/2))] ) )
            self.phase[:,i] = np.unwrap( np.angle( fft_inter[:,i]))
            
            # bin range for fitting the phase
            idx_start.append(int(self.pump_max_bin[i]-shift_fitphase))
            idx_end.append(int(self.pump_max_bin[i]+shift_fitphase))

            #linear fit to the phase
            phase_coeff.append( np.polyfit(np.arange( idx_start[i],idx_end[i] ),
                                           self.phase[idx_start[i]:idx_end[i],i],
                                           1))
            fitted_phase[:,i] = np.polyval( phase_coeff[i],
                                            np.arange(interf.shape[0]) )
            #phase correction (should always plot this)
            phasing_term[:,i] = np.exp(-1j*fitted_phase[:,i] )
            fft_inter[:,i] = fft_inter[:,i]*phasing_term[:,i]
            self.phase[:,i] = np.unwrap( np.angle( fft_inter[:,i]))
            
            if pump_corr == 1:
                for j in range(self.Npixels):
                    sig = np.roll(data[:,j,i],-int(self.binZero[i])) # shift to bin0
                    self.fft_sig[:,j,i] = np.fft.fft(sig)*phasing_term[:,i]/pump[:,i] #FFT
            elif pump_corr == 0:
                for j in range(self.Npixels):
                    sig = np.roll(data[:,j,i],-int(self.binZero[i])) # shift to bin0
                    self.fft_sig[:,j,i] = np.fft.fft(sig)*phasing_term[:,i] #FFT
            else:
                print('wrong input on pump_corr')
                pass

        print('binzero, phase',self.binZero)
        print('max of fft av interf',self.pump_max_bin)
        print('size of fft_inter',fft_inter.shape)
        print('size of fft_sig',self.fft_sig.shape)

        

            
    def plot_cuts(self, data, pump, wnums = [2050], mode = 'ver', save = 0, name = 'proj'):
        if mode == 'ver':
            ax = plot_snippet1D_init(w3_label, dabs_label)
            for j in range(self.lt):
                for i in wnums:
                    idx = np.argmin( abs(pump - i) )
                    plot1Dfrom2D(data[:,:,j],
                                 wnums,
                                 ax_to_cut=0, 
                                 ax0=pump, ax1=self.w, 
                                 const=0,
                                 offset=0,
                                 x0=0)
            plot_snippet1D_end(ax, save, name)
        if mode == 'hor':
            ib = np.argmin( abs(pump - self.w[0]) )
            ie = np.argmin( abs(pump - self.w[-1]) ) +1
            ax = plot_snippet1D_init(w3_label, dabs_label)
            for j in range(self.lt):
                for i in wnums:
                    idx = np.argmin( abs(self.w - i) )
                    plot1Dfrom2D(data[ib:ie,:,j],
                                 wnums,
                                 ax_to_cut=1, 
                                 ax0=pump[ib:ie], ax1=self.w, 
                                 const=0,
                                 offset=0,
                                 x0=0)
            plot_snippet1D_end(ax, save, name)



    def calib_probe(self, pump, spectrum = 0):
        ib = np.argmin( abs(pump - self.w[0]) ) -5
        ie = np.argmin( abs(pump - self.w[-1]) ) +5
        max_signal = []

        for i in range(32):
            max_signal.append(np.argmax(abs(self.fft_sig[ib:ie,i,spectrum])) )
            self.w[i] = pump[ib+max_signal[-1]]
        print(pump[ib:ie+1], max_signal)



    def plot_pp_proj(self,pump, save=0, name = 'pp_proj'):
        self.pp_proj = np.zeros((self.lt,32))
        ib = np.argmin( abs(pump - self.w[0]) )
        ie = np.argmin( abs(pump - self.w[-1]) ) +1 
        
        for i in range(self.lt):
            self.pp_proj[i,:] = np.sum(np.real(self.fft_sig[ib:ie,:,i]),axis=0)
            
        print(ib,ie)
        print(self.pp_proj.shape)
        ax = plot_snippet1D_init(w3_label, dabs_label)
        plot1Dfrom2D(self.pp_proj,
                     self.t,
                     ax_to_cut=0, 
                     ax0=self.t, ax1=self.w, 
                     const=0,
                     offset=0,
                     x0=0)
        plot_snippet1D_end(ax, save, name)

    def subtract_first_spectrum(self):
        self.data_sub_neg = self.fft_sig-self.fft_sig[:,:,0,None]



##################
##### script #####
##################
#root = tk.Tk()
#root.withdraw()

#extracting ftir
# os.chdir('/home/dpalec/Documents/UZH/Data/FTIR/180406')
# exlist = [f for f in os.listdir(os.getcwd()) if re.search("TiO2 Si Pt",f) and not re.search('~',f) and not re.search('.dat',f)]
# extract_opus(exlist)


# os.chdir('/home/dpalec/Documents/UZH/Data/FTIR/180426')
# listf = sorted([f for f in os.listdir(os.getcwd()) if re.search("TiO2 Pt on Si_",f) and not re.search('~',f)])

# d1 = FTIR(listf[:3])


# data = d1.get_data()
# d1.get_header()
# d1.get_dim()
# d1.glob_ref(1)

# d1.plot1D(save=1)
# d1.bcg_corr()
# d1.plot()
# d2 = FTIR(listf)
# d1.plot()
# d2.plot()

# d1.plot(list_idx = [1,2], save=0)
# d1.plot(list_idx = [1,2], corr=1, save=0)

# d1.rem_glob_ref()
# d1.plot(1,2)

# d1.bcg_corr()
# d1.svd(d1.data[30:55])

# d1.pca()
#d1 = FTIR()
#db.set_trace()
# ddir = '/home/dpalec/Documents/UZH/Data/2D/Water_h/20180528/pp_TiO2_Pt_KI_water_UV6_195444'
# pp = PP(ddir)
# #pp.plot(pp.avdata, t0 = 0, log=1,save=1)
# pp.def_offset(pp.avdata,method='auto')
# pp.plot_spectra(pp.avdata, popTimes = [0.2,0.5,2,5],save=0, offset = pp.offset)
# pp.fit_kin([2104],1,0.2,t_label, dabs_label, log=0,offset=pp.offset,save=1)
# pp.plot_kin(pp.avdata,wnums=[2104],offset=pp.offset,save=1)

#ddir = '/home/dpalec/Documents/UZH/Data/2D/mess_2d/test'
# ddir = '/home/dpalec/Documents/UZH/Data/2D/Water_h/20180528/2D_TiO2_Pt_KI_meoh_UV7_203411'
# md = MD(ddir)

# md.pre_phase()
# md.apodize()

# md.padding(md.avinterf, md.avsignal)
# md.phase(md.avsignal, md.avinterf,pump_corr=0)
# md.calib_probe(md.ax_pump)

# #md.fft_sig = md.fft_sig-md.fft_sig[:,:,0,None]

# #md.plot2D(md.apo_signal,md.ax_pump)
# md.plot2D(md.fft_sig,md.ax_pump,save=1)

# md.plot_pp_proj(md.ax_pump)
# #md.plot2D(md.fft_sig,md.ax_pump,save=1)
# md.plot_cuts(md.fft_sig, md.ax_pump, mode = 'hor')
