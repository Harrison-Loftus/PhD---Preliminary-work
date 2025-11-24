import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

start_time = time.time()

# constants
L = 1.0  # body length mm
R = 40.0e-3  # average body radius mm
r_c = 0.5e-3  # cuticle width mm
E_kPa = 1000 # young's modulus kPa
E = E_kPa * 1e-3 # convert to N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
k_b = E * I_c # bending viscosity N·mm^2
mu_b = 1.3e-7 # body viscocity N·mm²·s 
mu_f_mPas = np.array([1.0,10.0,1e2,1e3,2.8e4]) # fluid viscocity mPa·s
mu_f = mu_f_mPas * 1e-9 # N·s/mm^2
C_N = 3.4 * mu_f # normal drag coefficient N·s/mm^2
tao_b = mu_b / k_b # mechanical timescale seconds
tao_m = 100.0e-3 # muscle activation timescale seconds
tao_n = 10.0e-3 # neural activity timescale seconds
l = 1.0 / 6.0 # segment length


D_4 = np.array([[7, -4, 1, 0, 0, 0],
                [-4, 6, -4, 1, 0, 0],
                [1, -4, 6, -4, 1, 0],
                [0, 1, -4, 6, -4, 1],
                [0, 0, 1, -4, 6, -4],
                [0, 0, 0, 1, -4, 7]], float)

D_4 = D_4 * (1/(l**4))

W_p = np.array([[0,0,0,0,0,0],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0]], float)

W_g = np.array([[-1,1,0,0,0,0],
                [1,-2,1,0,0,0],
                [0,1,-2,1,0,0],
                [0,0,1,-2,1,0],
                [0,0,0,1,-2,1],
                [0,0,0,0,1,-1]], float)

I_6 = np.eye(6)

Kmat = -k_b * D_4 # precompute once


def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 2
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state, C_Nval):
    n = 6 # body segments
    kappa = state[0:n]

    A_V = state[n:2*n]
    A_D = state[2*n:3*n]
    V_V = state[3*n:4*n]
    V_D = state[4*n:5*n]
    
    epsilon_g = 0.0134
    epsilon_p = 0.05 # arbitrarily chosen to fit λ/L ≈ 1.6
    c_p = 1.0
    
    M = C_Nval * I_6 + mu_b * D_4
    dkappadt = np.linalg.solve(M, Kmat @ (kappa + (sigma(A_V) - sigma(A_D))))
    dA_Vdt = (1/tao_m)*(-A_V + V_V - V_D)
    dA_Ddt = (1/tao_m)*(-A_D + V_D - V_V) 

    dV_Vdt = (1/tao_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tao_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) # carter johnson's paper uses V_V as final term in eq 2.1, unsure if this is an error
    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results


T = 30
dt = 0.01
t_eval = np.arange(20, T, dt) # start time at 20s to avoid transients

# initial conditions
state = np.zeros(30)
state[18] = 1 # asymmetric perturbation in ventral and dorsal muscles
state[24] = -1
sols = []

for C_Nval in C_N:
    sol = solve_ivp(ODEs, [0, T], state, t_eval=t_eval, args=(C_Nval,), method='BDF', atol=1e-7, rtol=1e-6, max_step=0.05 )
    sols.append(sol)


for i in range(len(mu_f)):
    sol = sols[i]
    time1 = sol.t
    kappa = sol.y[0:6, :]

    plt.figure(figsize=(8,4))
    plt.imshow(kappa, aspect='auto', extent=[time1[0], time1[-1], 0, L], origin='lower', cmap='seismic')
    plt.colorbar(label='Curvature (1/mm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Body Length (mm)')
    plt.title(f'Kymograph of Curvature (μ_f = {mu_f_mPas[i]:.1e} mPa·s)')
    plt.tight_layout()
    plt.show()


print("--- %s seconds ---" % (time.time() - start_time))

# Kymograph plots of curvature for each fluid viscosity
# Initially, was unclear on whether a travelling wave would form in high viscosity fluids
# However, after adjusting the simulation time to be t in [20,30]s to avoid transients, travelling waves are visible