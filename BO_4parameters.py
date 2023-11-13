import GPy
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import time

from ocelot import *

def calc_resp(x0, x1, x2, x3):
    ''' calc beam optics for 2 QM
    x0 : QM1.k1
    x1 : QM2.k1
    '''
    # Define Optics
    ## Drift
    L20  = Drift(l=0.2, eid='L20')
    L50  = Drift(l=0.5, eid='L50')
    ## Quadrupoles
    QM1 = Quadrupole(l=0.2, k1=x0, eid='QM1')
    QM2 = Quadrupole(l=0.2, k1=x1, eid='QM2')
    QM3 = Quadrupole(l=0.2, k1=x2, eid='QM3')
    QM4 = Quadrupole(l=0.2, k1=x3, eid='QM4')
    ## Lattice
    cell = (L50, QM1, L20, QM2, L50, QM3, L20, QM4, L50)
    lat = MagneticLattice(cell, stop=None)
    ## Initial condition
    tws0 = Twiss()
    tws0.beta_x = 20.0
    tws0.beta_y = 30.0
    tws0.alpha_x = -10.0
    tws0.alpha_y = -5.0
    tws0.emit_x = 1.0
    tws0.emit_y = 1.0
    tws0.E = 1.0
    ## update optics
    lat.update_transfer_maps()
    tws = twiss(lat, tws0, nPoints=1000)

    idx = -1  # last point of the lattice
    sx = np.sqrt(tws0.emit_x * tws[idx].beta_x)
    sy = np.sqrt(tws0.emit_y * tws[idx].beta_y)

    return sx, sy, tws

def show_beta(x0, x1, x2, x3):
    sx, sy, tws = calc_resp(x0, x1, x2, x3)
    print(np.abs(sx),np.abs(sy))
    s = [p.s for p in tws]
    beta_x = [p.beta_x for p in tws]
    beta_y = [p.beta_y for p in tws]

    titlestr = "QM1.k1=%.2f, QM2.k1=%.2f,QM3.k1=%.2f, QM4.k1=%.2f, sx=%.3f, sy=%.3f"%(x0, x1, x2, x3, sx, sy)
    plt.plot(s, beta_x, 'b-');
    plt.plot(s, beta_y, 'r-');
    plt.legend(['Horiz', 'Vert'])
    plt.xlabel('s [m]')
    plt.ylabel('beta [m]')
    plt.title(titlestr)
    plt.grid(True)
    plt.show()

def eval_func(x):
    global cnt
    x0 = x[0][0]
    x1 = x[0][1]
    x2 = x[0][2]
    x3 = x[0][3]

    sx, sy, tws = calc_resp(x0, x1, x2, x3) # calc optics for 2 QM

    # Evaluation Function Examples
    val = np.abs(sx) + np.abs(sy)
    #val = np.abs(sx*sy)
    #val = np.log(np.abs((sx+10)*(sy+10)))
    #val = -1.0/np.abs((sx+10)*(sy+10))

    print("%d: %+8.3f,%+8.3f,%+8.3f,%+8.3f,%+8.3f,%+8.3f,%+8.3f"%(cnt, x0, x1, x2, x3, sx, sy, val))
    cnt = cnt + 1

    return val


# ==== Main =====
# GP Optimization
print("==== start ====")
print("# cnt, x0, x1, x2, x3, sx, sy, eval_val")
cnt = 0
bounds = [{'name':'x0', 'type':'continuous', 'domain':(-20, 20)},
          {'name':'x1', 'type':'continuous', 'domain':(-20, 20)},
          {'name':'x2', 'type':'continuous', 'domain':(-20, 20)},
          {'name':'x3', 'type':'continuous', 'domain':(-20, 20)}
          ]
'''
disc=np.arange(-20,20,0.1)
bounds = [{'name':'x0', 'type':'discrete', 'domain':disc},
          {'name':'x1', 'type':'discrete', 'domain':disc},
          {'name':'x2', 'type':'discrete', 'domain':disc},
          {'name':'x3', 'type':'discrete', 'domain':disc}
          ]
'''
myBopt = GPyOpt.methods.BayesianOptimization(f=eval_func,domain=bounds,initial_design_numdata=10,acquisition_type='LCB',acquisition_weight=2,de_duplication=True,normalize_Y=False,maximize=False)#,X=np.array(initial_X),Y=np.array(initial_Y))
#myBopt = GPyOpt.methods.BayesianOptimization(f=eval_func,domain=bounds,initial_design_numdata=10,acquisition_type='EI',jitter=0.01,de_duplication=True,normalize_Y=False,maximize=False)#,X=np.array(initial_X),Y=np.array(initial_Y))
#myBopt = GPyOpt.methods.BayesianOptimization(f=eval_func,domain=bounds,initial_design_numdata=10,acquisition_type='MPI',jitter=0.01,de_duplication=True,normalize_Y=False,maximize=False)#,X=np.array(initial_X),Y=np.array(initial_Y))

myBopt.run_optimization(max_iter=30)


#myBopt.plot_acquisition()
myBopt.plot_convergence()
print("Best     = ", myBopt.x_opt)

# plot betatron function
x0, x1, x2, x3 = np.array(myBopt.x_opt)

show_beta(x0, x1, x2, x3)
