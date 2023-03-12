import numpy as np
import matplotlib.pyplot as plt

with open('output_5000.txt', 'r') as serpent_file:
    k_eff_analog_list = []
    for line in serpent_file:
        if line.startswith("k-eff (analog)"):
            k_eff_analog_list.append(float(line[20:27]))

serpent_k_eff_array = np.asarray(k_eff_analog_list)

serpent_k_eff = np.average(serpent_k_eff_array)

with open('output_asmc.txt', 'r') as our_file:
    for line in our_file:
        if line.startswith("average k_eff"):
           our_k_eff = float(line[16:])

print(f'serpent k_eff = {serpent_k_eff}')
print(f'our k_eff = {our_k_eff}')
print(f'percent deviation = {abs(our_k_eff - serpent_k_eff) * 100 / serpent_k_eff}')

our_batch_size = 5000
our_cycles = 200

serpent_batch_size = 5000
serpent_cycles = 200

def calculate_power(batch_size, cycles):
    # assume that this number of cycles were completed in 1 second
    neutrons_per_sec = batch_size * cycles

    fissions_per_sec = neutrons_per_sec / 2.45

    power_eVps = fissions_per_sec * 200e6

    power_W = power_eVps * 1.6021766209e-19

    power_muW = power_W * 1e6
    
    return power_muW

print(f'serpent power = {calculate_power(serpent_batch_size, serpent_cycles)} micro-watts')
print(f'our power = {calculate_power(our_batch_size, our_cycles)} micro-watts')