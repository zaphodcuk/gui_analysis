import oop_plot_data as mc
import scipy.optimize as optim
import numpy as np


def get_entries(entries):
    init_vals = []
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        init_vals.append(float(entry[1].get()))
#        print('%s: "%s"' % (field, text))
        
    return init_vals

def residuals_2d(p,data,t,w_len,nexp):
    a_list = []
    t_list = []
    for i in range(nexp):
        a_list.append(p[data.shape[1]*i:data.shape[1]*i+data.shape[1]][:])
        t_list.append(p[i-nexp])
    res = data.ravel() - model_das(t,w_len,a_list,t_list,nexp)

    return (res)

def model_das(t,w_len,a_list,t_list,nexp):
    model = np.zeros((len(t),w_len))
    for i in range(w_len):
        for j in range(nexp):
            model[:,i] = model[:,i] + a_list[j][i]*np.exp(-t/t_list[j])
    return model.ravel()

def fit_exponentials(data,t,w,idx,nexp,tinit,log,t0=0):
    fit=[]
    bounds = mc.def_bounds(nexp)['b']
    p0 = mc.multiple_p0(nexp)
    iinit = np.argmin(abs(t-tinit))
    for i in range(idx):
        ff = []
        res = []
        for j in range(len(p0)):
            ff.append(optim.leastsq(mc.residuals,p0[j],
                                    args = (data[iinit:,idx],t[iinit:],nexp,bounds),
                                    maxfev=1000),)
            res.append(np.sum((data[iinit:,idx]- mc.func(t[iinit:]-t0,ff[j][0],nexp))**2))
        idx = np.argmin(res)
        fit.append(ff[idx])
#    aa.plot(t[iinit:]-t0,mc.func(t[iinit:]-t0,fit[0][0],nexp))
    return fit


def fit_das(data,t,nexp, a_list, tau_list,t0):
    p0 = []
    fit = []
    for i in range(nexp):
        p0 = p0 + list(a_list[i][:])

    for i in range(nexp):
        p0.append(tau_list[i])
        
    idx_init = np.argmin(abs(t-t0))
    
    fit.append(optim.leastsq(residuals_2d,p0,
                             args = (data[idx_init:,:],t[idx_init:],
                                     data.shape[1],nexp),maxfev=10000),)
    print('DAS fit goes from index:', idx_init)

    return fit

def fit_svd(dmatrix):
    X_std = StandardScaler().fit_transform(dmatrix)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    u,s,v = np.linalg.svd(X_std.T)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
