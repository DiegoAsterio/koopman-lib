import math
import random as rd 
from pathlib import Path
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pdb
import cProfile

# # miscelanea

# def fast_exp(x,n):
#     if n==0:
#         return 1
#     elif n == 1:
#         return x
#     elif n%2 == 0:
#         return fast_exp(x*x, n/2)
#     else:
#         return x*fast_exp(x*x, (n-1)/2)

# def linear_growth(lmbd):
#     fam = [lambda x : lmbd*x]
#     return get_obs(fam)

# def periodic_data(a0, aes, bs, T=1):
#     n = len(aes)
#     coss = [lambda x : a*np.cos(2*math.pi*i*x/T) for a,i in zip(aes,range(n))]
#     sins = [lambda x : b*np.sin(2*math.pi*i*x/T) for b,i in zip(bs,range(n))]
#     return get_obs([lambda x : a0/2] + coss + sins)

# def four_trig(a0, a1, b1, a2, b2, a3, b3, a4, b4):
#     return periodic_data(a0, [a1, a2, a3, a4], [b1, b2, b3, b4])

# I/O

def get_data(path_to_csvs):
    p = Path(path_to_csvs)
    all_files = p.glob('**/*.csv')

    # Generador O(n) en tiempo y O(1) en espacio
    df_each_file = (pd.read_csv(f) for f in all_files)
    data = pd.concat(df_each_file, ignore_index = True)
    data[' Timestamp'] = pd.to_datetime(data[' Timestamp'])
    return data

# def. time series

def get_time_series(data, time_var, selected_rows, time_step):
    selected_data = data[selected_rows]
    grouped_by_time = selected_data.groupby(time_var)
    time_series = grouped_by_time.size()
    return time_series.resample(time_step).sum()


# parametrized families of funct.

def get_obs(param_fam):
    eye = lambda x : x
    return [eye] + param_fam[:]
    
def mon(deg):
    return lambda x : x**deg

def polys(degs):
    fam = [mon(d) for d in degs]
    return get_obs(fam)

def four_polys(a, b, c, d):
    return polys([int(a), int(b), int(c), int(d)])

def harmonics(n,T):
    n = abs(int(n))
    T = abs(T)
    coss = [lambda x : np.cos(2*math.pi*n*x/T) for i in range(n)]
    sins = [lambda x : np.sin(2*math.pi*n*x/T) for i in range(n)]
    return get_obs([lambda x: 1] + coss + sins)

def exp_growth(b, mu):
    fam = [lambda x : b**(mu*x)]
    return get_obs(fam)

# koopman functs.

def koopman_op(Y):
    Y0 = Y[:,:-1]
    Y1 = Y[:,1:]
    return Y1 @ np.linalg.pinv(Y0)

def apply_koopman_particular_case(K, gs, x):
    col = np.array([[g(x)] for g in gs])
    return np.matmul(K[0],col)[0]

# prediction functs.

def prediction(xs, ds, gs):
    window = xs[:ds+1]
    Y = np.array([[g(x) for x in window] for g in gs])
    for xk in xs[ds+1:]:
        K = koopman_op(Y)
        p = [apply_koopman_particular_case(K, gs, x) for x in xs]
        yield p
        fresh = np.array([[g(xk)] for g in gs])
        Y = np.concatenate((Y[:,1:] , fresh), axis=1)

def cross_validation(xs, ds, gs, p=0.6):
    temp_series, pred = [], []
    while len(xs)>=ds:
        window, xs = xs[:ds+1], xs[ds+1:]
        index = int(len(window)*p)
        train, test = window[:index], window[index:]
        Y = np.array([[g(x) for x in train] for g in gs])
        K = koopman_op(Y)
        temp_series.append([x for x in test]+[x for x in xs[:1]])
        single_p = [apply_koopman_particular_case(K, gs, x) for x in test]
        pred.append(single_p)
    return temp_series, pred

def one_step_prediction(xs, ds, gs):
    window = xs[:ds+1]
    Y = np.array([[g(x) for x in window] for g in gs])
    K = koopman_op(Y)
    ret = [apply_koopman_particular_case(K, gs, x) for x in window]
    for xk in xs[ds+1:]:
        fresh = np.array([[g(xk)] for g in gs])
        Y = np.concatenate((Y[:,1:], fresh), axis=1)
        K = koopman_op(Y)
        ret.append(apply_koopman_particular_case(K, gs, xk))
    return ret
        
# estimation

def dai(x0,x1,xx1):
    if (xx1-x0)*(x1-x0) > 0:
        return 1
    elif (xx1-x0)*(x1-x0) < 0:
        return -1
    elif (xx1-x0)==(x1-x0):
        return 1
    else:
        return -1

def directio_accuracy(time_series, prediction):
    return np.array([ dai(x0,x1,xx1) for x0, x1, xx1 in zip(time_series, time_series[1:],prediction)])
        
def directio_fc_val(time_series, prediction):
    da = directio_accuracy(time_series, prediction)
    return np.array([abs(x1 - x0)*v for x0, x1, v in zip(time_series,time_series[1:],da)])

def mean_directio_acc(time_series, prediction):
    da = directio_accuracy(time_series, prediction)
    return np.mean(da)

def mean_directio_fc_val(time_series, prediction):
    dv = directio_fc_val(time_series, prediction)
    return np.mean(dv)

def norm_directio_fc_val(time_series, prediction):
    dv = directio_fc_val(time_series, prediction)
    s_gen = (abs(x1 - x0) for x0,x1 in zip(time_series, time_series[1:]))
    return (1+np.sum(dv))/(1+sum(s_gen))

def calculate_mmda(time_series, ps):
    return np.mean([mean_directio_acc(xs, p) for xs, p in zip(time_series,ps)])

def calculate_mmdv(time_series, ps):
    return np.mean([mean_directio_fc_val(xs, p) for xs, p in zip(time_series,ps)])

def calculate_nmdv(time_series, ps):
    return np.mean([norm_directio_fc_val(xs, p) for xs, p in zip(time_series,ps)])
    
def mmda_score(ts, window_cols, fam):
    xs, ps = cross_validation(ts, window_cols, fam)
    return calculate_mmda(xs, ps)

def mmdv_score(ts, window_cols, fam):
    xs, ps = cross_validation(ts, window_cols, fam)
    return calculate_mmdv(xs, ps)

def nmdv_score(ts, window_cols, fam):
    xs, ps = cross_validation(ts, window_cols, fam)
    return calculate_nmdv(xs, ps)

def q_mda_score(ts, window_cols, fam):
    p = one_step_prediction(ts, window_cols, fam)
    return mean_directio_acc(ts, p)

def q_mda_tuned(ts, window_cols, fam):
    score = q_mda_score(ts, window_cols, fam)
    return math.tan(0.5*math.pi*score)

def nmdv_tuned(ts, window_cols, fam):
    score = nmdv_score(ts, window_cols, fam)
    return math.tan(0.5*math.pi*score)

# Optimization problems

def estimate_gradient(funct, point, eps):
    n = len(point)
    gradient = []
    for i in range(n):
        delta = np.copy(point)
        delta[i] += eps
        tvm = (funct(point) - funct(delta))/eps
        gradient.append(tvm)
    return np.array(gradient)

def calc_learning_rate(x0, x1, g0, g1):
    dx = x0 - x1    
    dg = g0 - g1
    num = np.dot(dx, dg)
    den = np.sum(dg * dg)
    return num/den

# Steepest gradient

def objective(ts, window_cols, fam, param):
    fam_instance = fam(*param)
    return mmda_score(ts, window_cols, fam_instance)

def q_objective(ts, window_cols, fam, param):
    fam_instance = fam(*param)
    return q_mda_score(ts, window_cols, fam_instance)

def gradient_descent(objective, num_param, lr=0.1, eps=2.0):
    max_iter = 8000
    x1 = 10 * np.random.random(num_param)
    g1 = np.zeros(num_param)
    objs = []
    for i in range(max_iter):
        objs.append(objective(x1))
        g0, g1 = g1, estimate_gradient(objective, x1, eps)
        x0, x1 = x1, x1 - lr * g1
        if np.linalg.norm(x0-x1,ord=2) < 0.0001:
            return objs + [objective(x1)], objective(x1), x1
        if np.linalg.norm(g0-g1,ord=2) >= 0.0001:
            lr = calc_learning_rate(x0, x1, g0, g1)
    print("Did all iterations!\n")
    return objs, objective(x1), x1

# Simulated annealing

def ann_objective(ts, window_cols, param):
    n = len(param)
    degs = [i for x, i in zip(param, range(n)) if x]
    fam_instance = polys(degs)
    return mmda_score(ts, window_cols, fam_instance)

def prob(e0, e1, t):
    if e1 > e0:
        return 1
    else:
        return math.exp((e1 - e0)/t)

def rd_neighbour(p0):
    p1 = p0[:]
    n = len(p1)
    index = rd.randrange(0, n)
    p1[index] = not p1[index]
    return p1

def temperature(t0,t1,k):
    ratio = t1/t0
    return t0*ratio**k

def sim_ann(objective, num_param):
    p = [False for _ in range(num_param)]
    e = objective(p)
    max_iter = 10000
    t0 = 10000
    t1 = 9900
    objs, temps = [e], [t0]
    for k in range(max_iter):
        t = temperature(t0, t1, k)
        neigh = rd_neighbour(p)
        e_neigh = objective(neigh)
        if prob(e, e_neigh, t) >= rd.random():
            p = neigh
            e = e_neigh
        objs.append(e)
    return objs, temps, objective(p), p

# plot functs

def plot_cv(ts, ds, gs):
    xs, ps = cross_validation(ts, ds, gs)

    das = [1000*y for x, p in zip(xs,ps) for y in directio_accuracy(x, p)]

    axvs = [-1]
    for p in ps:
        axvs.append(axvs[-1]+len(p))
    axvs[0] = 0 
    # pdb.set_trace()
    xs = [y for x in xs for y in x[1:]]
    ps = [y for p in ps for y in p]
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.plot(xs, label="time_series")
    ax.plot(ps, label="prediction")
    ax.plot(das, 'ro', label="DA(i)", linestyle='')
    ax.vlines(axvs, ymin=0, ymax=1, transform=ax.get_xaxis_transform(), linestyles='dashed')
    plt.title("Cross validation prediction")
    plt.legend()
    plt.show()

def main():
    time_var=' Timestamp'

    data = get_data('Adware/')
    # Aqui se podria poner cualquier condicicion para seleccionar las filas
    all_rows = [True]*len(data)
    time_step = '10Min'
    adware_time_series = get_time_series(data, time_var, all_rows, time_step)

    ransom_data = get_data('Ransomware/')
    # Aqui se podria poner cualquier condicicion para seleccionar las filas
    ransom_all_rows = [True]*len(ransom_data)
    ransom_time_step = '10Min'
    ransom_time_series = get_time_series(ransom_data, time_var, ransom_all_rows, time_step)
    
    steps = [24, 48]
    tss = [adware_time_series, ransom_time_series]
    
    for time_series, name in zip(tss, ["Adware", "Ransomware"]):
        print("Predicting " + name + "\n")                
        for step in steps:
            print("Arbitrary harmonic dictionary\n")
            N, T = 8, 100
            gs = harmonics(N, T)
            score = mmda_score(time_series, step, gs)
    
            print("Objective function for harmonic dictionary of 2*{} elements over the period {}:\nWindow size:{} MDA:{:.4}\n".format(N, T, step, score))

            print("Tuned harmonic dictionary\n")

            f = lambda x: -objective(time_series, step, harmonics, x)
            score_list, g_score, params = gradient_descent(f, 2)

            print("Score after gradient ascent with 2*{}+1 elements and period {:.3}:\nWindow size:{} MDA:{:.4}\n".format(*params, step, -g_score))

            ys = [-y for y in score_list]

            fig, ax = plt.subplots()
            ax.plot(ys, label="MMDA")
            ax.plot([score]*len(ys), label="static MDA")
            plt.title("Gradient ascent")
            plt.legend()
            plt.show()

            gs = harmonics(*params)
                
            plot_cv(time_series, step, gs)
    
            print("Arbitrary polynomial dictionary\n")

            gs = four_polys(1.0, 2.0, 4.0, 5.0)
            score = mmda_score(time_series, step, gs)
            print("Original score:\nWindow size:{} MDA:{:.4}\n".format(step, score))

            print("Tuned polynomial dictionary\n")

            f = lambda x : ann_objective(time_series, step, x)
            score_list, temps, sa_score, main_degs = sim_ann(f, 6)
            print("Score after simulated annealing of the polynomial dictionary {} :\nWindow size:{} MDA:{:.4}\n".format(main_degs, step, sa_score))

            fig, ax = plt.subplots()
            ax.plot(score_list[2000], label="MDA")
            ax.plot([score]*len(2000), label="static MDA")
            plt.title("Simulated annealing")
            plt.legend()
            plt.show()
            
            degs = [i for x, i in zip(main_degs, range(len(main_degs))) if x]
            gs = polys(degs)

            plot_cv(time_series, step, gs)

    # Diccionarios originales del articulo
        D1 = [lambda x : x,
              lambda x : 1,
              lambda x : math.sin(x),
              lambda x : math.cos(x),
              lambda x : math.sin(2*x),
              lambda x : math.cos(2*x)]

        D2 = [lambda x : x,
              lambda x : 1,
              lambda x : x**2,
              lambda x : x**3,
              lambda x : x**4]

        D3 = [lambda x : x,
              lambda x : 1,
              lambda x : np.sin(x),
              lambda x : np.cos(x)]


        for s in steps:
            for D,i in zip([D1, D2, D3], range(1,4)):
                D_mmda = mmda_score(time_series, s, D)
                D_mmdv = mmdv_score(time_series, s, D)
                D_nmdv = nmdv_score(time_series, s, D)
                
                print("Score for original dictionary D{} with step {}\nMDA:{:.4}\tMDV:{:.4}\tNMDV:{:.4}\n".format(i, s, D_mmda, D_mmdv, D_nmdv))

                plot_cv(time_series, s, D)

if __name__ == "__main__":
    main()
