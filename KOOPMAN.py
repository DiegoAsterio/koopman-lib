import math
import random as rd 
from pathlib import Path
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pdb
import cProfile

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

# miscelanea

def fast_exp(x,n):
    if n==0:
        return 1
    elif n == 1:
        return x
    elif n%2 == 0:
        return fast_exp(x*x, n/2)
    else:
        return x*fast_exp(x*x, (n-1)/2)

# parametrized families of funct.

def get_obs(param_fam):
    eye = lambda x : x
    return [eye] + param_fam[:]
    
def mon(deg):
    return lambda x : fast_exp(x, deg)

def polys(degs):
    fam = [mon(d) for d in degs]
    return get_obs(fam)

def four_polys(a, b, c, d):
    return polys([int(a), int(b), int(c), int(d)])

def periodic_data(a0, aes, bs, T=1):
    n = len(aes)
    sins = [lambda x : a*np.sin(math.pi*n*x/T) for a,n in zip(aes,range(n))]
    coss = [lambda x : b*np.cos(math.pi*n*x/T) for b,n in zip(bs,range(n))]
    return get_obs([lambda x : a0] + sins + coss)

def four_trig(a0, a1, b1, a2, b2, a3, b3, a4, b4):
    return periodic_data(a0, [a1, a2, a3, a4], [b1, b2, b3, b4])

def linear_growth(lmbd):
    fam = [lambda x : lmbd*x]
    return get_obs(fam)

def exp_growth(b, mu):
    fam = [lambda x : b**(mu*x)]
    return get_obs(fam)

# koopman functs.

# def koopman_operator(time_series, gs):
#     Y = np.array([[g(x) for x in time_series] for g in gs])
#     Y0 = Y[:,:-1]
#     Y1 = Y[:,1:]
#     return Y1 @ np.linalg.pinv(Y0)

def koopman_op(Y):
    Y0 = Y[:,:-1]
    Y1 = Y[:,1:]
    return Y1 @ np.linalg.pinv(Y0)

def apply_koopman_particular_case(K, gs, x):
    row = np.array([g(x) for g in gs])
    return(np.matmul(K[0],row.transpose()))

# prediction functs.

def prediction(time_series, gs, step):
    window = time_series[0: step]
    Y = np.array([[g(x) for x in window] for g in gs])
    for xk in time_series[step:]:
        refresher = np.array([[g(xk)] for g in gs])
        Y = np.concatenate((Y, refresher), axis=1)
        K = koopman_op(Y)
        p = [apply_koopman_particular_case(K, gs, x) for x in time_series]
        yield p
        Y = Y[:,1:]

def one_step_prediction(xs, gs, d):
    window = time_series[0 : d + 1]
    Y = np.array([[g(x) for x in window] for g in gs])
    K = koopman_op(Y)
    ret = [apply_koopman_particular_case(K, gs, x) for x in window]
    for xk in time_series[d+1:]:
        fresh = np.array([[g(xk)] for g in gs])
        Y = np.concatenate((Y[:,1:], fresh), axis=1)
        K = koopman_op(Y)
        ret.append(apply_koopman_particular_case(K, gs, xk))
    return ret
        
# estimation

def directio_accuracy(time_series, prediction):
    return np.array([ 1 if (xx1-x0)*(x1-x0) > 0 else -1 for x0, x1, xx1 in zip(time_series, time_series[1:],prediction)])
        
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
    return np.sum(dv)/sum(s_gen)

def calculate_mmda(time_series, ps):
    return np.mean([mean_directio_acc(time_series, p) for p in ps])

def calculate_nmdv(time_series, ps):
    return np.mean([norm_directio_fc_val(time_series, p) for p in ps])
    
def mmda_score(ts, window_cols, fam):
    ps = prediction(ts, fam, window_cols)
    return calculate_mmda(ts, ps)

def q_mda_score(ts, window_cols, fam):
    p = one_step_prediction(ts, fam, window_cols)
    return mean_directio_acc(ts, p)

def nmdv_score(ts, window_cols, fam):
    ps = prediction(ts, fam, window_cols)
    return calculate_nmdv(ts, ps)

def nmdv_tuned(ts, window_cols, fam):
    score = nmdv_score(ts, window_cols, fam)
    return math.tan(0.5*math.pi*score)

# Optimization problems

# TODO: revisar
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
    return nmdv_tuned(ts, window_cols, fam_instance)

def q_objective(ts, window_cols, fam, param):
    fam_instance = fam(*param)
    p = one_step_prediction(ts, fam_instance, window_cols)
    print(p[30:35], "\n")
    return mean_directio_acc(ts, p)

def gradient_descent(objective, num_param, lr=0.1, eps=2.5):
    max_iter = 1000
    x1 = 30 * np.random.random(num_param)
    g1 = np.zeros(num_param)
    objs = []
    for i in range(max_iter):
        objs.append(objective(x1))
        g0, g1 = g1, estimate_gradient(objective, x1, eps)
        x0, x1 = x1, x1 - lr * g1
        if np.linalg.norm(x0-x1,ord=2) < 0.0001:
            return objs, objective(x1)
        lr = calc_learning_rate(x0, x1, g0, g1)
    return objs, objective(x1)

# Simulated annealing

def ann_objective(ts, window_cols, param):
    n = len(param)
    degs = [i for x, i in zip(param, range(n)) if x]
    fam_instance = polys(degs)
    return nmdv_tuned(ts, window_cols, fam_instance)

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
    p1 = [False for _ in range(num_param)]
    max_iter = 1000
    t0 = 10000
    t1 = 9000
    objs = []
    for k in range(max_iter):
        t = temperature(t0, t1, k)
        p0, p1 = p1, rd_neighbour(p1)
        e0, e1 = objective(p0), objective(p1)
        objs.append(e0)
        if prob(e0, e1, t) < rd.random():
            p1 = p0
    return objs, objective(p1)

if __name__ == "__main__":
    data = get_data('Adware/')

    time_var=' Timestamp'
    all_rows = np.array([True for _ in range(len(data))])
    time_step = '10Min'
    time_series = get_time_series(data, time_var, all_rows, time_step)
    # toy_time_series = time_series[:100]
    step = 30

    # print("Trigonometric dictionary\n")

    # gs = four_trig(1.0, 5.0, 8.0, 2.0, 1.0, 5.0, 7.0, 5.0, 6.0)
    # score = q_mda_score(time_series, step, gs)
    # print("Original score: %f\n" % score)
    
    # f = lambda x : -q_objective(time_series, step, four_trig, x)
    
    # score_list, g_score = gradient_descent(f, 9)
    # print("Score after gradient_ascent: %f\n" % g_score)

    print("Linear dictionary\n")

    gs = linear_growth(7.0)
    score = q_mda_score(time_series, step, gs)
    print("Original score: %f\n" % score)

    f = lambda x : - q_objective(time_series, step, linear_growth, x)
    score_list, g_score = gradient_descent(f, 1)
    print("Score after gradient ascent: %f\n" % g_score)
    
    # print("Polynomial dictionary\n")

    # gs = four_polys(1.0, 2.0, 4.0, 5.0)
    # score = nmdv_tuned(toy_time_series, step, gs)
    # print("Original score: %f\n" % score)

    # f = lambda x : ann_objective(toy_time_series, step, x)
    # score_list, sa_score = sim_ann(f, 6)
    # print("Score after simulated annealing: %f\n" % sa_score)

    # plt.plot(score_list)
    # plt.ylabel("mda")
    # plt.show()

