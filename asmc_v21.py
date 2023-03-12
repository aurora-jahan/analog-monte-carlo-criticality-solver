import numpy as np
import matplotlib.pyplot as plt
import asmc_module_v20 as aur
# import plotly.graph_objects as go
from sys import exit, stdout
import time
from multiprocessing import Pool


# MAKE THE Interactions

fission = aur.interaction('fission')
capture = aur.interaction('capture')
scattering = aur.interaction('scattering')

interaction_types = [fission, capture, scattering]

print('interactions created')

# cross section file path
xs_path = 'XS_DATA/'

# MAKE THE NUCLIDES

U235 = aur.nuclide('92235.09c', 235, 2.4355, xs_path + 'U235_total_ENDF8.csv', xs_path + 'U235_fission_ENDF8.csv', xs_path + 'U235_capture_ENDF8.csv')
U238 = aur.nuclide('92238.09c', 238, 2.819, xs_path + 'U238_total_ENDF8.csv', xs_path + 'U238_fission_ENDF8.csv', xs_path + 'U238_capture_ENDF8.csv')
C12 = aur.nuclide('6012.09c', 12, 0, xs_path + 'C12_total_ENDF8.csv', xs_path + 'C12_fission_ENDF8.csv', xs_path + 'C12_capture_ENDF8.csv')
H1 = aur.nuclide('1001.09c', 1, 0, xs_path + 'H1_total_ENDF8.csv', xs_path + 'H1_fission_ENDF8.csv', xs_path + 'H1_capture_ENDF8.csv')
O16 = aur.nuclide('8016.09c', 16, 0, xs_path + 'O16_total_ENDF8.csv', xs_path + 'O16_fission_ENDF8.csv', xs_path + 'O16_capture_ENDF8.csv')

print('nuclides created')

# MAKE THE MATERIAL

Fuel = aur.material('Fuel', [U235, U238, C12], [2.748e+20, 4.662e+21, 8.807e+23])
Water = aur.material('Water', [H1, O16], [6.692e22, 3.346e22])
Outside = aur.material('Outside', [], [])

print('materials created')


# MAKE THE GEOMETRY

# create some surfaces
cyl_1 = aur.cylindrical_surface_par2z(1, 250, 250, 0, 50, 100, 400)
box_2 = aur.box_surface(2, 100, 400, 100, 400, 0, 500)
surface_list = [cyl_1, box_2]

print('surfaces created')

# create regions according to the surfaces
region_1 = aur.region(3, [-1, -2], Fuel)
region_2 = aur.region(2, [1, -2], Water)
region_3 = aur.region(1, [1, 2], Outside)
region_list = [region_1, region_2, region_3]

print('regions created')

# define the overall boundaries of the system
x_min = 0
x_max = 500
y_min = 0
y_max = 500
z_min = 0
z_max = 500

# show the geometry
aur.show_geometry(10, x_min, x_max, y_min, y_max, z_min, z_max, surface_list, region_list)
print('geometry plot created')

# CREATE A BATCH OF INITIAL NEUTRONS
batch_size = 5000
print(f'\nbatch size = {batch_size}')
neutron_list = []
    
for n in range(0, batch_size):
    neutron_list.append(aur.generate_neutron_in_core(x_min, x_max, y_min, y_max, z_min, z_max, surface_list, region_list))
    # print("\r", end="")
    # print(f'{n + 1} neutrons created', end="")


# CYCLES
cycles = 200
print(f'\nnumber of cycles = {cycles}')

# list to store neutron population sizes for k_eff calculation
neutron_population = [len(neutron_list)]

# list to store k_eff values
k_eff_list = []

# array to store flux values
step_size = 1
x_boxes = np.arange(x_min, x_max + step_size, step_size)
y_boxes = np.arange(y_min, y_max + step_size, step_size)
z_boxes = np.arange(z_min, z_max + step_size, step_size)
total_flux_array = np.zeros((len(x_boxes), len(y_boxes), len(z_boxes)))


# Multiprocessing
processes_allowed = 12
print(f'\nprocesses allowed = {processes_allowed}')

# create a pool of processes
process_pool = Pool(processes=processes_allowed)

# make the intervals to split up the neutron list for multiprocessing
interval_list = []
for i in range(0, processes_allowed):
    interval_list.append(int(batch_size * i / processes_allowed))
interval_list.append(batch_size - 1)

# list to store time data for measuring multirocessing performance
t = []
t.append(time.perf_counter())

# SIMULATE NEUTRON HISTORIES
print(f'\nsimulating {batch_size} neutrons for {cycles} cycles using {processes_allowed} processes...')

# iterate through the cycles
for cycle in range(0, cycles):

    print(f'\ncycle {cycle + 1}:')

    results_list = []
    
    for i in range(0, len(interval_list) - 1):
        args_i = (batch_size, neutron_list[interval_list[i]:interval_list[i + 1]], 
                  interaction_types, surface_list, region_list, x_boxes, y_boxes, z_boxes)
        results_list.append(process_pool.apply_async(aur.simulate_neutron_histories, args_i))
    
    t.append(time.perf_counter())
    
    # clear neutron_list to store new batch of neutrons
    neutron_list = []
    
    # get the results
    for result in results_list:
        total_flux_array += result.get()[0]
        neutron_list += result.get()[1]
    
    # calculate k_eff
    neutron_population.append(len(neutron_list))
    k_eff = neutron_population[cycle + 1] / neutron_population[cycle]
    print(f'\nk_eff = {k_eff}')
    k_eff_list.append(k_eff)
    
    # normalize the neutron population
    if len(neutron_list) > batch_size:
        while len(neutron_list) > batch_size:
            to_eliminate = np.random.randint(low=0, high=len(neutron_list))
            neutron_list.remove(neutron_list[to_eliminate])
    elif len(neutron_list) < batch_size:
        while len(neutron_list) < batch_size:
            neutron_list.append(np.random.default_rng().choice(neutron_list))
    else:
        pass
    
    t.append(time.perf_counter())
    
# close the process pool
process_pool.close()
process_pool.join()

# calculate average k_eff   
k_eff_array = np.asarray(k_eff_list)
print(f'\naverage k_eff = {np.average(k_eff_array)}')

# show a 2D visualization of neutron flux integrated along the z axis
flux_2d_array = np.sum(total_flux_array, axis=2)
plt.pcolormesh(flux_2d_array, cmap='inferno')
plt.savefig('2d_flux.png', bbox_inches='tight')
# plt.show()

# assess multithreaded performance
t = np.asarray(t)

start = t[0]
end = t[-1]
print(f'total execution time = {end - start}')

t = t - start
plt.plot(t)
plt.savefig('time.png', bbox_inches='tight')
# plt.show()