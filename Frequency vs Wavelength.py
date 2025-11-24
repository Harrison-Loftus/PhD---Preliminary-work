import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
from scipy.signal import find_peaks
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
t_eval = np.arange(20, T, dt)

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
    kappa = sol.y[0:6, :]  # 6 curvature segments

    
wave_lengths = []
for i in range(len(mu_f)):
    sol = sols[i]
    kappa = sol.y[0:6, :]
    phase_1 = np.unwrap(np.angle(hilbert(kappa[0,:])))
    phase_2 = np.unwrap(np.angle(hilbert(kappa[1,:])))
    phase_3 = np.unwrap(np.angle(hilbert(kappa[2,:])))
    phase_4 = np.unwrap(np.angle(hilbert(kappa[3,:])))
    phase_5 = np.unwrap(np.angle(hilbert(kappa[4,:])))
    phase_6 = np.unwrap(np.angle(hilbert(kappa[5,:])))


    phi_1 = np.mean(((phase_2 - phase_1) % (2*np.pi)) / (2*np.pi))
    phi_2 = np.mean(((phase_3 - phase_2) % (2*np.pi)) / (2*np.pi))
    phi_3 = np.mean(((phase_4 - phase_3) % (2*np.pi)) / (2*np.pi))
    phi_4 = np.mean(((phase_5 - phase_4) % (2*np.pi)) / (2*np.pi))
    phi_5 = np.mean(((phase_6 - phase_5) % (2*np.pi)) / (2*np.pi))

    phi = np.array([phi_1, phi_2, phi_3, phi_4, phi_5])
    lam_over_L = (1/6) * 5 / np.sum(1 - phi) # as per appendix A.1.3 carter johnson
    
    wave_lengths.append(lam_over_L)
    

frequencies = []
for i in range(len(mu_f)):
    sol = sols[i]
    kappa = sol.y[0, :]
    time1 = sol.t

    peaks, _ = find_peaks(kappa, height=0.2, prominence=0.4)
    peak_times = time1[peaks]
    periods = np.diff(peak_times)
    average_period = np.mean(periods)
    frequency = 1 / average_period

    frequencies.append(frequency)


plt.figure(figsize=(10,4))
plt.plot(mu_f_mPas, frequencies, marker='o', color='k')
plt.xscale('log')
plt.xlabel('Fluid Viscosity (mPa·s)', fontsize=14)
plt.ylabel('Frequency (Hz)', fontsize=14)
plt.title('Frequency vs Fluid Viscosity', fontsize=16)
plt.show()


plt.figure(figsize=(10,4))
plt.plot(wave_lengths, frequencies, marker='o', color='k')
plt.xlabel('Normalised Wavelength', fontsize=14)
plt.ylabel('Frequency (Hz)', fontsize=14)
plt.title('Frequency vs Normalised Wavelength', fontsize=16)
plt.show()


print("--- %s seconds ---" % (time.time() - start_time))

# Currently our model does not show the same trend as experimental data for frequency vs fluid viscosity.
# In experiments, as fluid viscosity increases, frequency decreases. ~1.7 - 0.3Hz Fang et al
# vs ~1.53 - 1.48Hz in our model.