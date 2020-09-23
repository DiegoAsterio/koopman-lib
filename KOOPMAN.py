import math
import random
from pathlib import Path
import copy

import numpy as np
import pandas as pd

import pdb

# I/O

def get_data(path_to_csvs):
    p = Path(path_to_csvs)
    all_files = p.glob('**/*.csv')

    df_each_file = (pd.read_csv(f) for f in all_files)
    data = pd.concat(df_each_file, ignore_index = True)
    data[' Timestamp'] = pd.to_datetime(data[' Timestamp'])
    return data

# def. time series

def get_time_series(data, time_var, selected_rows, time_step):
    selected_data = data[selected_rows]
    grouped_by_time = selected_data.groupby(time_var)
    time_series = grouped_by_time.size()
    #time_series.index = pd.to_datetime(time_series.index)
    return time_series.resample(time_step).sum()

# miscelanea

def quick_pow(x,n):
    if n==0:
        return 1
    p = quick_pow(x, n//2)
    if (n%2 == 0):
        return p*p
    else:
        return n*p*p

# parametrized families of funct.

def get_observables(param_fam):
    eye = lambda x : x
    # pdb.set_trace()
    return [eye] + copy.deepcopy(param_fam)
    
def mon(deg):
    return lambda x : quick_pow(x, deg)

def get_p_vect_obs(degs):
    fam = [mon(d) for d in degs]
    return get_observables(fam)

def four_polys(a, b, c, d):
    # pdb.set_trace()
    return get_p_vect_obs([int(a)%8, int(b)%8, int(c)%8, int(d)%8])

def periodic_data(n, theta):
    fam = [lambda x : np.sin(2*math.pi*m*x/theta) for m in range(1,math.floor(n)+1)]
    return get_observables(fam)

def linear_growth(lmbd):
    fam = [lambda x : lmbd*x]
    return get_observables(fam)

def exp_growth(b, mu):
    fam = [lambda x : b**(mu*x)]
    return get_observables(fam)

# koopman functs.

def koopman_operator(time_series, gs):
    Y = np.array([[g(x) for x in time_series] for g in gs])
    Y0 = Y[:,:-1]
    Y1 = Y[:,1:]
    return Y1 @ np.linalg.pinv(Y0)

def apply_koopman_particular_case(K, gs, x):
    row = np.array([g(x) for g in gs])
    return(np.matmul(K[0],row.transpose()))

# prediction functs.

def strange_prediction(time_series, gs, step):
    init = 0
    n = len(time_series)
    predictions = []
    while init + step + 1 <= n:
        window = time_series[init : init + step + 1]
        # pdb.set_trace()
        K = koopman_operator(window, gs)
        p = [apply_koopman_particular_case(K, gs, x) for x in time_series]
        yield p
        init += 1

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
    ps = strange_prediction(ts, fam, window_cols)
    return calculate_mmda(ts, ps)

def nmdv_score(ts, window_cols, fam):
    ps = strange_prediction(ts, fam, window_cols)
    return calculate_nmdv(ts, ps)

def nmdv_tuned(ts, window_cols, fam):
    score = nmdv_score(ts, window_cols, fam)
    return math.tan(score)

# Optimization problems

# TODO: revisar
def estimate_gradient(funct, point, eps):
    n = len(point)
    gradient = []
    for i in range(n):
        delta = np.copy(point)
        delta[i] += eps
        # pdb.set_trace()
        tvm = (funct(point) - funct(delta))/eps
        # pdb.set_trace()
        gradient.append(tvm)
    return np.array(gradient)

# TODO: revisar
def calc_learning_rate(x0, x1, g0, g1):
    dx = x0 - x1    
    dg = g0 - g1
    num = np.dot(dx, dg)
    den = np.sum(dg * dg)
    # pdb.set_trace()
    return num/den

# TODO: revisar
def objective(ts, window_cols, fam, param):
    fam_instance = fam(*param)
    # pdb.set_trace()
    return nmdv_tuned(ts, window_cols, fam_instance)


def gradient_ascent(objective, num_param, lr=0.1, eps=1.0):
    max_iter = 1000
    x1 = 8 * np.random.random(num_param)
    g1 = np.zeros(num_param)
    for i in range(max_iter):
        # pdb.set_trace()
        g0, g1 = g1, estimate_gradient(objective, x1, eps)
        x0, x1 = x1, x1 + lr * g1
        # pdb.set_trace()
        lr = calc_learning_rate(x0, x1, g0, g1)
        if i == 100:
            print("The 100th iteration gave score %f\n" % objective(x1))
    return objective(x1)

if __name__ == "__main__":
    data = get_data('Adware/')

    time_var=' Timestamp'
    all_rows = np.array([True for _ in range(len(data))])
    time_step = '10Min'
    time_series = get_time_series(data, time_var, all_rows, time_step)
    toy_time_series = time_series[:100]
    step = 30

    trig_fam = periodic_data(5.0, 8.0)
    gs = get_observables(trig_fam)

    score = mmda_score(toy_time_series, step, trig_fam)
    print("Original score: %f\n" % score)
    
    f = lambda x : objective(toy_time_series, step, periodic_data, x)
    g_score = gradient_ascent(f, 2)
    print("Score after gradient_ascent: %f\n" % g_score)
