import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import plotly.graph_objects as go
import fileinput

def display_distribution_of_RN(x, n=10):
    intervals = np.linspace(np.amin(x), np.amax(x), n + 1)
    points_in_intervals = np.zeros(n)
    
    points_in_intervals, intervals = np.histogram(x, bins=intervals)

    plt.plot(intervals[0:(n - 1)], points_in_intervals[0:(n - 1)], 'r')
    plt.xlabel('Variable (unit)')
    plt.ylabel('number of samples')
    plt.ylim(bottom=np.amin(points_in_intervals) - 0.2 * np.amin(points_in_intervals))
    plt.ylim(top=np.amax(points_in_intervals) + 0.2 * np.amax(points_in_intervals))
    plt.show()

# the pdf
def pdf(x):
    p_E_W = 0.484 * np.exp(- x) * np.sinh(np.sqrt(2 * x))
    return p_E_W

# the line pdf
def line_pdf(x, x1, y1, x2, y2):
    m = (y1 - y2) / (x1 - x2)
    c = y1 - x1 * (y1 - y2) / (x1 - x2)
    h_x =  m * x + c
    return h_x

# the line cdf
def line_cdf(x, x1, y1, x2, y2):
    m = (y1 - y2) / (x1 - x2)
    c = y1 - x1 * (y1 - y2) / (x1 - x2)
    F_x =  m * x**2 / 2 + c * x
    return F_x

# inverse of the line cdf
def inv_line_cdf(x, x1, y1, x2, y2):
    m = (y1 - y2) / (x1 - x2)
    c = y1 - x1 * (y1 - y2) / (x1 - x2)
    F_inv_x = - c / m + (np.sqrt(c**2 + 2 * m * x))/m
    return F_inv_x

# Acceptance rejection method using triangle approach
def sample_energy_from_Watt_spectrum():
    
    while True:
        uniform_rn = np.random.rand()

        prob_scaled_rn_1 = inv_line_cdf(uniform_rn, 0, 0.1, 20, 0)

        c = 4
        h = line_pdf(prob_scaled_rn_1, 0, 0.1, 20, 0)
        u = np.random.rand()
        f = pdf(prob_scaled_rn_1)
        
        if u * c * h <= f:
            E = prob_scaled_rn_1
            break
        else:
            continue
    
    return E


def show_neutrons_3D(neutron_list, dataset=None, color_map="PiYG"):
    # Creating dataset
    
    x = np.zeros(len(neutron_list))
    for p in range(0, len(neutron_list)):
        x[p] = neutron_list[p].location_x
    
    y = np.zeros(len(neutron_list))
    for q in range(0, len(neutron_list)):
        y[q] = neutron_list[q].location_y
    
    z = np.zeros(len(neutron_list))
    for r in range(0, len(neutron_list)):
        z[r] = neutron_list[r].location_z
    
    e = np.zeros(len(neutron_list))
    if dataset == None:
        for r in range(0, len(neutron_list)):
            e[r] = neutron_list[r].energy
    else:
        e = dataset
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, c=e, cmap=color_map)
    plt.title("Neutrons in a box")
    
    # show plot
    plt.show()
    
#direction of flight of the neutron 
def sample_random_direction_of_flight():
    omega_z = np.random.default_rng().uniform(-1, 1)
    phi = np.random.default_rng().uniform(0, 2 * np.pi)
    omega_x = np.cos(phi) * np.sqrt(1 - omega_z**2)
    omega_y = np.sin(phi) * np.sqrt(1 - omega_z**2)
    
    return omega_x, omega_y, omega_z

# microscopic cross sections of nuclides
def make_sigma(xs_file):
    sigma_vs_E = pd.read_csv(xs_file, sep=';')       # reading the csv file from Janis
    sigma_vs_E = sigma_vs_E.to_numpy()      # transforming the pandas dataframe into a numpy array

    sigma_vs_E[:, 0] = sigma_vs_E[:, 0] * 1e-6      # turning E in eV from Janis into E in MeV for our use case
    sigma_vs_E[:, 1] = sigma_vs_E[:, 1] * 1e-24     # turning sigma in barns from Janis into sigma in cm^2 for our use case
    
    # SIGMA_vs_E = np.empty_like(sigma_vs_E)
    # SIGMA_vs_E[:, 0] = sigma_vs_E[:, 0]                 # energy
    # SIGMA_vs_E[:, 1] = sigma_vs_E[:, 1] * atom_density  # SIGMA = N * sigma
    
    #for multiple materials, use SIGMA = (N * sigma)_1 + (N * sigma)_2 +.... _n
    
    return sigma_vs_E

# macroscopic cross sections of materials
def make_SIGMA(sigma_vs_E, atom_density):
    SIGMA_vs_E = np.empty_like(sigma_vs_E)
    SIGMA_vs_E[:, 0] = sigma_vs_E[:, 0]                 # energy
    SIGMA_vs_E[:, 1] = sigma_vs_E[:, 1] * atom_density  # SIGMA = N * sigma
    
    #for multiple materials, use SIGMA = (N * sigma)_1 + (N * sigma)_2 +.... _n
    
    return SIGMA_vs_E

def calculate_SIGMAs_at_nE(n, mat):
    # calculate the macroscopic cross-sections of all the nuclides at this neutron's energy
    SIGMA_t = np.zeros(len(mat.nuclides))
    SIGMA_f = np.zeros(len(mat.nuclides))
    SIGMA_c = np.zeros(len(mat.nuclides))
    SIGMA_s = np.zeros(len(mat.nuclides))
    
    for i in range(0, len(SIGMA_t)):
        SIGMA_t[i] = np.interp(n.energy, mat.SIGMA_ts_vs_E[i][:, 0], mat.SIGMA_ts_vs_E[i][:, 1])
        
    for i in range(0, len(SIGMA_f)):
        SIGMA_f[i] = np.interp(n.energy, mat.SIGMA_fs_vs_E[i][:, 0], mat.SIGMA_fs_vs_E[i][:, 1])
    
    for i in range(0, len(SIGMA_c)):
        SIGMA_c[i] = np.interp(n.energy, mat.SIGMA_cs_vs_E[i][:, 0], mat.SIGMA_cs_vs_E[i][:, 1])
    
    for i in range(0, len(SIGMA_s)):
        SIGMA_s[i] = np.interp(n.energy, mat.SIGMA_ss_vs_E[i][:, 0], mat.SIGMA_ss_vs_E[i][:, 1])
        
    return SIGMA_t, SIGMA_f, SIGMA_c, SIGMA_s
        
def calculate_SIGMA_t_at_nE(n, mat):
    # calculate the macroscopic cross-sections of all the nuclides at this neutron's energy
    SIGMA_t = np.zeros(len(mat.nuclides))
    
    for i in range(0, len(SIGMA_t)):
        SIGMA_t[i] = np.interp(n.energy, mat.SIGMA_ts_vs_E[i][:, 0], mat.SIGMA_ts_vs_E[i][:, 1])
    
    return SIGMA_t

def calculate_largest_SIGMA_t_at_nE(n, region_list):
    SIGMA_t_regions = np.zeros(len(region_list))
    
    for j in range(0, len(SIGMA_t_regions)):
        current_region = region_list[j]
        mat = current_region.material
        SIGMA_t_nucs = np.zeros(len(mat.nuclides))
        
        for i in range(0, len(SIGMA_t_nucs)):
            SIGMA_t_nucs[i] = np.interp(n.energy, mat.SIGMA_ts_vs_E[i][:, 0], mat.SIGMA_ts_vs_E[i][:, 1])
        
        SIGMA_t_regions[j] = np.sum(SIGMA_t_nucs)
    
    largest_SIGMA_t = np.amax(SIGMA_t_regions)
    
    return largest_SIGMA_t

    
    
def sample_collision_distance(SIGMA_t):
    # calculate total macroscopic calculation of the whole material
    SIGMA_t_mat_at_neutron_energy = np.sum(SIGMA_t)
    
    # sample random distance to the next collision
    s = (- 1 / SIGMA_t_mat_at_neutron_energy) * np.log(np.random.rand())
    
    return s

def sample_nuclide(SIGMA_t, mat):
    
    SIGMA_t_mat_at_neutron_energy = np.sum(SIGMA_t)
    
    # calculate the probabilities
    p_i_array = SIGMA_t / SIGMA_t_mat_at_neutron_energy
    
    # create the options
    a_i = []
    for i in range(0, len(p_i_array)):
        a_i.append(i)
    
    # choose nuclide
    colliding_with_index = np.random.default_rng().choice(a=a_i, p=p_i_array)
    colliding_with_nuclide = mat.nuclides[colliding_with_index]
    
    return colliding_with_index, colliding_with_nuclide

def sample_interaction(SIGMA_t, SIGMA_f, SIGMA_c, colliding_with_index):
    
    # calculate the probabilities
    prob_fission = SIGMA_f[colliding_with_index] / SIGMA_t[colliding_with_index]
    prob_capture = SIGMA_c[colliding_with_index] / SIGMA_t[colliding_with_index]
    prob_scattering = 1 - prob_fission - prob_capture
    p_j_array = np.array([prob_fission, prob_capture, prob_scattering])
    
    # create the options
    a_j = []
    for j in range(0, len(p_j_array)):
        a_j.append(j)
    
    # choose interaction type
    interaction_type_index = np.random.default_rng().choice(a=a_j, p=p_j_array)

    return interaction_type_index


# interaction objects
class interaction:
    def __init__(self, name):
        self.name = name
        
# nuclide objects

class nuclide:
    def __init__(self, name, mass_number, nubar, total_xs, fission_xs, capture_xs):
        self.name = name
        self.A = mass_number
        self.nubar = nubar
        
        total_xs_array = make_sigma(total_xs)
        fission_xs_array = make_sigma(fission_xs)
        capture_xs_array = make_sigma(capture_xs)
        energy_array = total_xs_array[0, :]
        scattering_xs = total_xs_array[1, :] - fission_xs_array[1, :] - capture_xs_array[1, :]
        scattering_xs_array = np.array([energy_array, scattering_xs])
        
        self.sigma_t_vs_E = total_xs_array
        self.sigma_f_vs_E = fission_xs_array
        self.sigma_c_vs_E = capture_xs_array
        self.sigma_s_vs_E = scattering_xs_array
        
        # self.sigma_s_vs_E = aur.make_SIGMA(scattering_xs, atomic_concentration)
    

# material objects 

class material:
    def __init__(self, name, nuclides, atomic_concentrations):
        self.name = name
        self.nuclides = nuclides
        self.atomic_concentrations = atomic_concentrations
        
        SIGMA_ts = []
        for i in range(0, len(nuclides)):
            SIGMA_ts.append(make_SIGMA(nuclides[i].sigma_t_vs_E, atomic_concentrations[i]))
        self.SIGMA_ts_vs_E = SIGMA_ts
        
        SIGMA_fs = []
        for i in range(0, len(nuclides)):
            SIGMA_fs.append(make_SIGMA(nuclides[i].sigma_f_vs_E, atomic_concentrations[i]))
        self.SIGMA_fs_vs_E = SIGMA_fs
        
        SIGMA_cs = []
        for i in range(0, len(nuclides)):
            SIGMA_cs.append(make_SIGMA(nuclides[i].sigma_c_vs_E, atomic_concentrations[i]))
        self.SIGMA_cs_vs_E = SIGMA_cs
        
        SIGMA_ss = []
        for i in range(0, len(nuclides)):
            SIGMA_ss.append(make_SIGMA(nuclides[i].sigma_s_vs_E, atomic_concentrations[i]))
        self.SIGMA_ss_vs_E = SIGMA_ss           


# neutron objects

class neutron:
  def __init__(self, x, y, z, w_x, w_y, w_z, energy, alive):
    self.location_x = x
    self.location_y = y
    self.location_z = z
    self.omega_x = w_x
    self.omega_y = w_y
    self.omega_z = w_z
    self.energy = energy
    self.alive = alive


def generate_neutron_randomly(x_min, x_max, y_min, y_max, z_min, z_max):
    n_location_x = np.random.default_rng().uniform(x_min, x_max)
    n_location_y = np.random.default_rng().uniform(y_min, y_max)
    n_location_z = np.random.default_rng().uniform(z_min, z_max)
    
    # n_location_x = (x_max - x_min) / 2
    # n_location_y = (y_max - y_min) / 2
    # n_location_z = (z_max - z_min) / 2
    
    n_energy = sample_energy_from_Watt_spectrum()
    
    omega_x, omega_y, omega_z = sample_random_direction_of_flight()
    
    new_neutron = neutron(n_location_x, n_location_y, n_location_z, omega_x, omega_y, omega_z, n_energy, True)
    
    return new_neutron

def generate_neutron_at_center(x_min, x_max, y_min, y_max, z_min, z_max):
    # n_location_x = np.random.default_rng().uniform(x_min, x_max)
    # n_location_y = np.random.default_rng().uniform(y_min, y_max)
    # n_location_z = np.random.default_rng().uniform(z_min, z_max)
    
    n_location_x = (x_max - x_min) / 2
    n_location_y = (y_max - y_min) / 2
    n_location_z = (z_max - z_min) / 2
    
    n_energy = sample_energy_from_Watt_spectrum()
    
    omega_x, omega_y, omega_z = sample_random_direction_of_flight()
    
    new_neutron = neutron(n_location_x, n_location_y, n_location_z, omega_x, omega_y, omega_z, n_energy, True)
    
    return new_neutron

def generate_neutron_at_location(x, y, z):
    # n_location_x = np.random.default_rng().uniform(x_min, x_max)
    # n_location_y = np.random.default_rng().uniform(y_min, y_max)
    # n_location_z = np.random.default_rng().uniform(z_min, z_max)
    
    n_location_x = x
    n_location_y = y
    n_location_z = z
    
    n_energy = sample_energy_from_Watt_spectrum()
    
    omega_x, omega_y, omega_z = sample_random_direction_of_flight()
    
    new_neutron = neutron(n_location_x, n_location_y, n_location_z, omega_x, omega_y, omega_z, n_energy, True)
    
    return new_neutron

def generate_neutron_in_core(x_min, x_max, y_min, y_max, z_min, z_max, surface_list, region_list):
    while True:
        x = np.random.default_rng().uniform(x_min, x_max)
        y = np.random.default_rng().uniform(y_min, y_max)
        z = np.random.default_rng().uniform(z_min, z_max)
        
        material_at_xyz = check_material_xyz(x, y, z, surface_list, region_list)

        # if material_at_xyz.name == 'Outside':
        #     continue
        if material_at_xyz.name == 'Fuel':
            n_location_x = x
            n_location_y = y
            n_location_z = z
            break
        else:
            continue
    
    n_energy = sample_energy_from_Watt_spectrum()
    
    omega_x, omega_y, omega_z = sample_random_direction_of_flight()
    
    new_neutron = neutron(n_location_x, n_location_y, n_location_z, omega_x, omega_y, omega_z, n_energy, True)
    
    return new_neutron

def check_material_xyz(x, y, z, surface_list, region_list):
    surface_senses = []
    
    for a_surface in surface_list:
        surface_senses.append(a_surface.check_sense(x, y, z))
         
    for a_region in region_list:
        if surface_senses == a_region.signature:
            return a_region.material
        else:
            pass

def check_material(n, surface_list, region_list):
    surface_senses = []
    
    for a_surface in surface_list:
        surface_senses.append(a_surface.check_sense(n.location_x, n.location_y, n.location_z))
        
    for a_region in region_list:
        if surface_senses == a_region.signature:
            return a_region.material
        else:
            pass

class region:
    def __init__(self, number, signature, material=None):
        self.number = number
        self.signature = signature
        self.material = material

class plane_surface:
    def __init__(self, number, a, b, c, d):
        self.number = number
        self.a = a
        self.b = b
        self.c = c
        self.d = -d
        
    def check_sense(self, x, y, z):
        surface_eqn_LHS = self.a * x + self.b * y + self.c * z + self.d
        if surface_eqn_LHS > 0:
            return self.number
        elif surface_eqn_LHS <= 0:
            return -self.number

class cylindrical_surface_par2x:
    def __init__(self, number, a, b, c, R, low, high):
        self.number = number
        self.a = a
        self.b = b
        self.c = c
        self.R = R
        self.high_surface = plane_surface(994, 1, 0, 0, high)
        self.low_surface = plane_surface(995, 1, 0, 0, low)
        
    def check_sense(self, x, y, z):
        surface_eqn_LHS = (y - self.b)**2 + (z - self.c)**2 - self.R**2
        if surface_eqn_LHS <= 0 and self.high_surface.check_sense(x, y, z) < 0 and self.low_surface.check_sense(x, y, z) > 0:
            return -self.number
        else:
            return self.number
        
class cylindrical_surface_par2y:
    def __init__(self, number, a, b, c, R, low, high):
        self.number = number
        self.a = a
        self.b = b
        self.c = c
        self.R = R
        self.high_surface = plane_surface(996, 0, 1, 0, high)
        self.low_surface = plane_surface(997, 0, 1, 0, low)
        
    def check_sense(self, x, y, z):
        surface_eqn_LHS = (x - self.a)**2 + (z - self.c)**2 - self.R**2
        if surface_eqn_LHS <= 0 and self.high_surface.check_sense(x, y, z) < 0 and self.low_surface.check_sense(x, y, z) > 0:
            return -self.number
        else:
            return self.number
        
class cylindrical_surface_par2z:
    def __init__(self, number, a, b, c, R, low, high):
        self.number = number
        self.a = a
        self.b = b
        self.c = c
        self.R = R
        self.high_surface = plane_surface(999, 0, 0, 1, high)
        self.low_surface = plane_surface(998, 0, 0, 1, low)
        
    def check_sense(self, x, y, z):
        surface_eqn_LHS = (x - self.a)**2 + (y - self.b)**2 - self.R**2
        if surface_eqn_LHS <= 0 and self.high_surface.check_sense(x, y, z) < 0 and self.low_surface.check_sense(x, y, z) > 0:
            return -self.number
        else:
            return self.number

class box_surface:
    def __init__(self, number, low_x, high_x, low_y, high_y, low_z, high_z):
        self.number = number
        self.low_x_surface = plane_surface(300, 1, 0, 0, low_x)
        self.high_x_surface = plane_surface(301, 1, 0, 0, high_x)
        self.low_y_surface = plane_surface(302, 0, 1, 0, low_y)
        self.high_y_surface = plane_surface(303, 0, 1, 0, high_y)
        self.low_z_surface = plane_surface(304, 0, 0, 1, low_z)
        self.high_z_surface = plane_surface(305, 0, 0, 1, high_z)
        
    def check_sense(self, x, y, z):
        low_x_sense = self.low_x_surface.check_sense(x, y, z)
        high_x_sense = self.high_x_surface.check_sense(x, y, z)
        low_y_sense = self.low_y_surface.check_sense(x, y, z)
        high_y_sense = self.high_y_surface.check_sense(x, y, z)
        low_z_sense = self.low_z_surface.check_sense(x, y, z)
        high_z_sense = self.high_z_surface.check_sense(x, y, z)
        if low_x_sense > 0 and high_x_sense < 0 and low_y_sense > 0 and high_y_sense < 0 and low_z_sense > 0 and high_z_sense < 0:
            return -self.number
        else:
            return self.number
        
# show geometry
def show_geometry(step_size, x_min, x_max, y_min, y_max, z_min, z_max, surface_list, region_list):
    x = np.arange(x_min, x_max + step_size, step_size)
    y = np.arange(y_min, y_max + step_size, step_size)
    z = np.arange(z_min, z_max + step_size, step_size)

    regions_of_existence = np.zeros((len(x), len(y), len(z)))

    for k in range(0, len(z)):
        for j in range(0, len(y)):
            for i in range(0, len(x)):
                
                surface_senses = []
        
                for a_surface in surface_list:
                    surface_senses.append(a_surface.check_sense(x[i], y[j], z[k]))
                    
                for a_region in region_list:
                    if surface_senses == a_region.signature:
                        regions_of_existence[i, j, k] = a_region.number
                    else:
                        pass

    
    # plt.pcolormesh(regions_of_existence[:, :, 11])
    # plt.show()
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z)

    fig = go.Figure(data=go.Volume(
        x=X_grid.flatten(),
        y=Y_grid.flatten(),
        z=Z_grid.flatten(),
        value=regions_of_existence.flatten(),
        isomin=np.amin(regions_of_existence),
        isomax=np.amax(regions_of_existence),
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    fig.write_html("geometry.html")

    
def simulate_neutron_histories(batch_size, neutron_list, interaction_types, 
                               surface_list, region_list, x_boxes, y_boxes, z_boxes):
    
    # new flux array for this cycle
    flux_array = np.zeros((len(x_boxes), len(y_boxes), len(z_boxes)))
    
    # list to store the new neutrons born in this cycle
    new_neutron_list = []
    
    neutron_number = 0
    
    # iterate through the batch
    for n in neutron_list:
        
        neutron_number += 1
        
        # print("\r", end="")
        # print(f'simulating neutron {neutron_number} of {batch_size}... ', end="")
        
        # check which region we are in at the starting position
        mat = check_material(n, surface_list, region_list)
        
        # continue to simulate the neutron history until the neutron is dead
        while n.alive == True:
            
            # calculate the SIGMA_t at the current energy
            SIGMA_t = calculate_SIGMA_t_at_nE(n, mat)
            
            # calculate the largest SIGMA_t at the current energy
            largest_SIGMA_t = calculate_largest_SIGMA_t_at_nE(n, region_list)
            
            # calculate the appropriate SIGMA_v for this region
            SIGMA_v = largest_SIGMA_t - np.sum(SIGMA_t)
            
            # sample random distance to the next collision using the largest SIGMA_t
            s = sample_collision_distance(largest_SIGMA_t)
            
            # flux calculation
            x_box = np.nonzero(x_boxes < n.location_x)[0][-1]
            y_box = np.nonzero(y_boxes < n.location_y)[0][-1]
            z_box = np.nonzero(z_boxes < n.location_z)[0][-1]
            
            flux_array[x_box, y_box, z_box] += s
            
            # move to new position
            n.location_x = n.location_x + s * n.omega_x
            n.location_y = n.location_y + s * n.omega_y
            n.location_z = n.location_z + s * n.omega_z
            
            # check material at this new position
            mat = check_material(n, surface_list, region_list)
            
            # check for leakage, kill neutrons if out of bounds
            if mat.name == 'Outside':
                cause_of_death = 'leakage'
                n.alive = False
                print('leaked')
                continue
            
            # check if virtual or real
            prob_virtual = SIGMA_v / largest_SIGMA_t
            prob_real = np.sum(SIGMA_t) / largest_SIGMA_t
            virtual_or_real = np.random.default_rng().choice(a=['virtual', 'real'], p=[prob_virtual, prob_real])
            
            if virtual_or_real == 'virtual':
                # do nothing
                continue    # begin next while loop
            else:
                pass    # continue with this while loop
            
            # calculate the macroscopic cross-sections of all the nuclides at this neutron's energy
            SIGMA_t, SIGMA_f, SIGMA_c, SIGMA_s = calculate_SIGMAs_at_nE(n, mat)
            
            # sample nuclide that the neutron collides with
            colliding_with_index, colliding_with_nuclide = sample_nuclide(SIGMA_t, mat)
        
            # sample interaction type
            interaction_type_index = sample_interaction(SIGMA_t, SIGMA_f, SIGMA_c, colliding_with_index)
            interaction_type = interaction_types[interaction_type_index]

            if interaction_type_index == 0:
                # fission, so kill the neutron
                n.alive = False
                cause_of_death = 'fission'
                
                # calculate probabilities for fission neutron number
                fission_neutrons_high = np.ceil(colliding_with_nuclide.nubar)
                fission_neutrons_low = np.floor(colliding_with_nuclide.nubar)
                prob_low = fission_neutrons_high - colliding_with_nuclide.nubar
                prob_high = colliding_with_nuclide.nubar - fission_neutrons_low
                
                # calculate the number of new neutrons to be born
                fission_neutrons = np.random.default_rng().choice(a=np.array([fission_neutrons_low, fission_neutrons_high]), p=np.array([prob_low, prob_high]))

                # create the new neutrons
                for i in range(0, int(fission_neutrons)):
                    new_neutron_list.append(generate_neutron_at_location(n.location_x, n.location_y, n.location_z))
                
            if interaction_type_index == 1:
                # capture, so kill the neutron
                n.alive = False
                cause_of_death = 'capture'
            
            if interaction_type_index == 2:
                # scattering
                
                # pick direction randomly in COM system
                omega_c_x, omega_c_y, omega_c_z = sample_random_direction_of_flight()
                
                # calculate mu_c which is the cosine of the scattering angle in COM system
                mu_c = omega_c_x * n.omega_x + omega_c_y * n.omega_y + omega_c_z * n.omega_z   # dot product
                
                # calculate the size of the new direction vector, omega_new = A * omega_c + omega_old
                omega_size_sq = colliding_with_nuclide.A**2 + 2 * colliding_with_nuclide.A * mu_c + 1

                # calculate the components of the new direction vector
                omega_x_new = (colliding_with_nuclide.A * omega_c_x + n.omega_x) / np.sqrt(omega_size_sq)
                omega_y_new = (colliding_with_nuclide.A * omega_c_y + n.omega_y) / np.sqrt(omega_size_sq)
                omega_z_new = (colliding_with_nuclide.A * omega_c_z + n.omega_z) / np.sqrt(omega_size_sq)
                
                # assign new direction to the neutron
                n.omega_x = omega_x_new 
                n.omega_y = omega_y_new
                n.omega_z = omega_z_new
                
                # calculate new energy
                new_energy = omega_size_sq * n.energy / (colliding_with_nuclide.A + 1)**2
                
                # assign new energy to the neutron
                n.energy = new_energy
        
        # print(cause_of_death, end="")
    
    # # replace dead neutrons with newly born neutrons
    # neutron_list = new_neutron_list
    # new_neutron_list = []
    
    # # calculate k_eff
    # neutron_population.append(len(neutron_list))
    # k_eff = neutron_population[cycle + 1] / neutron_population[cycle]
    # print(f'\nk_eff = {k_eff}')
    # # k_eff_list.append(k_eff)
    
    # # normalize the neutron population
    # if len(neutron_list) > batch_size:
    #     while len(neutron_list) > batch_size:
    #         to_eliminate = np.random.randint(low=0, high=len(neutron_list))
    #         neutron_list.remove(neutron_list[to_eliminate])
    # elif len(neutron_list) < batch_size:
    #     while len(neutron_list) < batch_size:
    #         neutron_list.append(np.random.default_rng().choice(neutron_list))
    # else:
    #     pass
    
    # return k_eff, flux_array, neutron_list, new_neutron_list
    # return flux_array, neutron_list, new_neutron_list
    return flux_array, new_neutron_list