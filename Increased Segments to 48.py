import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
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
tau_b = mu_b / k_b # mechanical timescale seconds
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds
n = 48 # number of body segments
l = 1.0 / n # segment length 1 / n


D_4 = np.zeros((n, n), float)
for i in range(n):
    for j in range(n):
        if i == j:
            if i == 0 or i == n-1:
                D_4[i, j] = 7
            else:
                D_4[i, j] = 6
        elif abs(i - j) == 1:
            if i == 0 or i == n-1:
                D_4[i, j] = -4
            else:
                D_4[i, j] = -4
        elif abs(i - j) == 2:
            if i == 0 or i == n-1:
                D_4[i, j] = 1
            else:
                D_4[i, j] = 1


D_4 = D_4 * (1/(l**4))

W_p = np.zeros((n, n), float)
for i in range(1, n):
    W_p[i, i-1] = 1


W_g = np.zeros((n, n), float)
for i in range(n):
    for j in range(n):
        if i == j:
            if i == 0 or i == n-1:
                W_g[i, j] = -1
            else:
                W_g[i, j] = -2
        elif abs(i - j) == 1:
            W_g[i, j] = 1


I_6 = np.eye(n)

Kmat = -k_b * D_4 # precompute once


def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 2
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state, C_Nval):

    kappa = state[0:n]

    A_V = state[n:2*n]
    A_D = state[2*n:3*n]
    V_V = state[3*n:4*n]
    V_D = state[4*n:5*n]
    
    epsilon_g = 0.0134
    epsilon_p = 0.015625 # arbitrarily chosen to fit λ/L ≈ 1.6, as per bisection method
    c_p = 1
    
    M = C_Nval * I_6 + mu_b * D_4
    dkappadt = np.linalg.solve(M, Kmat @ (kappa + (sigma(A_V) - sigma(A_D))))
    dA_Vdt = (1/tau_m)*(-A_V + V_V - V_D)
    dA_Ddt = (1/tau_m)*(-A_D + V_D - V_V) 


    dV_Vdt = (1/tau_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) # carter johnson's paper uses V_V as final term in eq 2.1, unsure if this is an error

    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results


T = 30
dt = 0.01
t_eval = np.arange(20, T, dt) # start time at 20s to avoid transients

# initial conditions
state = np.zeros(5*n)
state[3*n] = 1 # asymmetric perturbation in ventral and dorsal muscles
state[4*n] = -1
sols = []

for C_Nval in C_N:
    sol = solve_ivp(ODEs, [0, T], state, t_eval=t_eval, args=(C_Nval,), method='Radau', atol=1e-7, rtol=1e-6, max_step=0.05 )
    sols.append(sol)

mid = n // 2
for i in range(len(mu_f)): 
    sol = sols[i]
    time1 = sol.t
    kappa = sol.y[0:n, :]  # n curvature segments

    plt.figure(figsize=(8, 5))
    for j in range(mid - 3, mid + 3):
        plt.plot(time1, kappa[j, :], label=f"Module {j+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Curvature κ (1/mm)")
    plt.title(f"Curvature vs Time for μ_f = {mu_f_mPas[i]:.1e} mPa·s")
    plt.legend()
    plt.show()

    
wave_lengths = []

for i in range(len(mu_f)):
    sol = sols[i]
    kappa = sol.y[0:n, :]            

    # compute instantaneous phase per segment
    phase_segments = []
    for j in range(n):
        analytic_signal = hilbert(kappa[j, :])
        instantaneous_phase = np.angle(analytic_signal)
        phase_segments.append(instantaneous_phase)

    # compute mean phase difference (phi_j) between adjacent segments
    phi_j_list = []
    for j in range(n-1):
        
        phase_diff = np.unwrap(phase_segments[j+1]) - np.unwrap(phase_segments[j])
        phi_j = np.mean(phase_diff) % (2*np.pi) / (2*np.pi)    # mean phase difference in cycles
        phi_j_list.append(phi_j)

    # convert to numpy array and compute lambda/L
    phi_arr = np.array(phi_j_list)
    lam_over_L = ((n-1)/n) / np.sum(1.0 - phi_arr)   
    wave_lengths.append(lam_over_L)
    

plt.figure(figsize=(10, 4))
plt.plot(mu_f_mPas, wave_lengths, marker='o', color='k')
plt.xscale('log')
plt.xlabel("Fluid Viscosity μ_f (mPa·s)")
plt.ylabel("Normalised Wavelength λ/L")
plt.title("Normalised Wavelength vs Fluid Viscosity")
plt.show()

                 
print("--- %s seconds ---" % (time.time() - start_time))