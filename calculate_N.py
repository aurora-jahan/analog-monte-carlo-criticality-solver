import math
import numpy

N_A = 6.022e23
M_235 = 235.0439
M_238 = 238.0508
M_C = 12
# M_O = 15.999
rho_mix = 19.5

w_U235 = 0.0055
w_U238 = 0.0945
w_C = 0.9

# M_U = 1 / (w_U235/M_235 + w_U238/M_238)

# M_UO2 = M_U + 2 * M_O

# rho_U = rho_UO2 * M_U / 

rho_U235 = rho_mix * w_U235
rho_U238 = rho_mix * w_U238
rho_C = rho_mix * w_C

N_U235 = rho_U235 * N_A / M_235
N_U238 = rho_U238 * N_A / M_238
N_C = rho_C * N_A / M_C

print(N_C, N_U235, N_U238)