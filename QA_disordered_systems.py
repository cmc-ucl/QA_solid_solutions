#!/usr/bin/env python
# coding: utf-8

# # QA disordered systems

# - [Multiplicity](#multiplicity)
# - [Build the structures](#structures)
# - [Build the test/train set](#manual_confcount)
# - [Build the QUBO model](#ml_energies)
# - [Build H$_{\text{ryd}}$ from pbc calculations](#h_ryd)
# - [Run the anneal](#anneal_binary)
#     - [GaAlN](#anneal_binary_gaaln) 
#     - [WMo](#anneal_binary_wmo) 
# - [Test QUBO classical](#test_QUBO_classical)
# - [Experiments](#experiments)

# In[3]:


from quantum_computing_functions import *
from quantum_computing_postprocessing import *

from dwave.embedding.chain_strength import  uniform_torque_compensation

import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import *

from ase.visualize import view

from pymatgen.ext.matproj import MPRester

from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
import sys
sys.path.insert(1,'/Users/brunocamino/Desktop/Imperial/crystal-code-tools/CRYSTALpytools/CRYSTALpytools/')
from crystal_io import *
from convert import *
import re
import shutil as sh

#from CRYSTALpytools.crystal_io import * 
#from CRYSTALpytools.convert import * 

import copy
from sklearn.metrics import mean_squared_error 

import dataframe_image as dfi
#from dscribe.descriptors import CoulombMatrix
from scipy import constants

import matplotlib.pyplot as plt

import itertools
from itertools import chain

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from dwave.system import EmbeddingComposite, DWaveSampler
import dimod 

k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]
# print(k_b)
def vview(structure):
    view(AseAtomsAdaptor().get_atoms(structure))

np.seterr(divide='ignore')
plt.style.use('tableau-colorblind10')

import seaborn as sns


# In[3]:


import sys
sys.path.insert(1,'/Users/brunocamino/Desktop/Imperial/crystal-code-tools/CRYSTALpytools/CRYSTALpytools/')
from crystal_io import *
from convert import *


# In[45]:


def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):
    """
    Calculate the partition function and probabilities for different energy levels.
    
    Args:
        energy (np.ndarray): Array of energy levels.
        multiplicity (np.ndarray): Array of corresponding multiplicities.
        T (float, optional): Temperature in Kelvin. Default is 298.15 K.
        return_pi (bool, optional): Flag to return probabilities. Default is True.
        N_N (float, optional): Number of N particles. Default is 0.
        N_potential (float, optional): Potential for N particles. Default is 0.

    Returns:
        tuple or float: If return_pi is True, returns a tuple containing partition function and probabilities.
                        Otherwise, returns the partition function.
    """
    k_b = 8.617333262145e-05  # Boltzmann constant in eV/K
    
    energy = np.array(energy)
    multiplicity = np.array(multiplicity)
    p_i = multiplicity * np.exp((-energy + (N_N * N_potential)) / (k_b * T))
    
    pf = np.sum(p_i)
    
    p_i /= pf
    
    if return_pi:
        return pf, p_i
    else:
        return pf


# In[44]:


def generate_symm_irr_structures(gui_object, output_object, new_atom, atom_type1 = False):
    """
    Generate symmetrically-irreducible structures with atom substitutions.

    Args:
        gui_object: The GUI object.
        output_object: The output object.
        new_atom: The new atom to substitute.

    Returns:
        List of generated structures.
    """
    output_object.get_config_analysis(return_multiplicity=True)
    original_structure = cry_gui2pmg(gui_object)
    multiplicity = output_object.multiplicity
    structures = []

    if atom_type1 == False:
        for substitutions in output_object.atom_type2:
            new_structure = original_structure.copy()
            for i in substitutions:
                new_structure.replace(i - 1, new_atom)
            structures.append(new_structure)
    else:
        for substitutions in output_object.atom_type1:
            new_structure = original_structure.copy()
            for i in substitutions:
                new_structure.replace(i - 1, new_atom)
            structures.append(new_structure)
    

    return structures, multiplicity


# In[43]:


#THIS WORKS WITH CRYSTAL
def get_all_configurations(gui_object):

    symmops = np.array(gui_object.symmops)
    coordinates = np.array(gui_object.atom_positions)
    n_symmops = gui_object.n_symmops
    atom_numbers = np.array(gui_object.atom_number)
    lattice = gui_object.lattice
    
    original_structure = Structure(lattice,atom_numbers,coordinates,coords_are_cartesian=True)

        
        
    rotations = []
    translation = []
    for symmop in symmops:
        rotations.append(symmop[0:3])
        translation.append(symmop[3:4])
    atom_indices = []
    structures = []
    for i in range(n_symmops):
        atom_indices_tmp = []
        coordinates_new = np.matmul(coordinates,rotations[i])+np.tile(translation[i], (len(atom_numbers),1))

        #lattice_new = np.matmul(lattice,rotations[i])+np.tile(translation[i], (3,1))
        structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=True)
        for k,coord in enumerate(original_structure.frac_coords):
            structure_tmp.append(original_structure.atomic_numbers[k],coord,coords_are_cartesian=False,validate_proximity=False)
        for m in range(len(atom_numbers)):
            index = len(atom_numbers)+m
            for n in range(len(atom_numbers)):
                if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index]):
                    #print(m,n)
                    atom_indices_tmp.append(n)
                    break
        atom_indices.append(atom_indices_tmp)

    return atom_indices
#atom_indices = get_all_configurations(gui_object)


# In[42]:


#THIS WORKS WITH PYMATGEN
def get_all_configurations_pmg(structure_pmg,prec=6):

    symmops = SpacegroupAnalyzer(structure_pmg).get_symmetry_operations()
#     print(len(symmops))
#     print(symmops)
    coordinates = np.round(np.array(structure_pmg.frac_coords),prec)
    n_symmops = len(symmops)
    atom_numbers = np.array(structure_pmg.atomic_numbers)
    lattice = structure_pmg.lattice.matrix
    
    original_structure_pmg = copy.deepcopy(structure_pmg)
            
    rotations = []
    translation = []
#     for i in range(n_symmops):
#         rotations.append(symmop[i].rotation_matrix)
#         translation.append(symmop[i].translation_vector)
    atom_indices = np.ones((len(symmops),structure_pmg.num_sites),dtype='int')
    atom_indices *= -1
#     print(atom_indices.shape)
    structures = []
    for i,symmop in enumerate(symmops):
#         print(symmop.rotation_matrix)
        atom_indices_tmp = []
        coordinates_new = []
        for site in coordinates:
            coordinates_new.append(np.round(symmop.operate(site),prec))
#         print('n',len(coordinates_new))
        structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=False,
                                  to_unit_cell=False)
#         if i == 1:
#             vview(structure_tmp)
        for k,coord in enumerate(original_structure_pmg.frac_coords):
            structure_tmp.append(original_structure_pmg.atomic_numbers[k],coord,coords_are_cartesian=False,
                                 validate_proximity=False)
        
#         if i ==1:
# #             print(structure_tmp.num_sites)
# #              structure_tmp.translate_sites(np.arange(structure_tmp.num_sites),[-1,-1,-1])
#             print('HEREE',structure_tmp.num_sites)
# #             return (structure_tmp)
        for m in range(len(atom_numbers)):
            index = len(atom_numbers)+m
            for n in range(len(atom_numbers)):
#                 if i ==1:
#                     print(m,n,structure_tmp.frac_coords[n]-structure_tmp.frac_coords[index],
#                           structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index]))
                    
#                 print(m,n,structure_tmp.frac_coords[n]-structure_tmp.frac_coords[index],structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index]))
                if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index],tolerance=0.001):
                    
#                     atom_indices_tmp.append(n)
                    atom_indices[i,m] = n
                    break
#         atom_indices[m] = (atom_indices_tmp)

    return atom_indices

#Build all configurations (complete set)
def build_test_train_set_from_bv(bva,energies_train,atom_indices):
    all_configurations = []
    all_energies = []
    #energies = list(chain(*graphene_allN_cry_energy_norm[1:max_N]))
    
    for i,bv in enumerate(bva):
        
        N_index = np.where(bv==1)[0]
        
        all_configurations.extend(build_symmetry_equivalent_configurations(atom_indices,N_index).tolist())
        all_energies.extend([energies_train[i]]*len(build_symmetry_equivalent_configurations(atom_indices,N_index)))
        #print(i,len(all_configurations))
    all_configurations = np.array(all_configurations)
    all_energies = np.array(all_energies)
    
    return all_configurations, all_energies
# In[41]:


def build_symmetry_equivalent_configurations(atom_indices,N_index):
    
    if len(N_index) == 0:
        #return np.tile(np.zeros(len(atom_indices[0]),dtype='int'), (len(atom_indices), 1))
        return np.array([np.zeros(len(atom_indices[0]),dtype='int')]) # TEST
    configurations = atom_indices == -1
    #print(configurations)
    for index in N_index:
        configurations += atom_indices == index
    configurations = configurations.astype(int)

    unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
    return unique_configurations

#     structures_new = []
#     for structure_index in unique_configurations_index:
#         structure_tmp = copy.deepcopy(structures[structure_index])
#         for index in N_index:
#             structure_tmp.replace(index,7)
#         structures_new.append(structure_tmp)


# In[40]:


#Build all configurations (complete set)
def build_test_train_set(structures_train,energies_train,atom_indices,N_atom):
    all_configurations = []
    all_energies = []
    #energies = list(chain(*graphene_allN_cry_energy_norm[1:max_N]))
    
    for i,structure in enumerate(structures_train):
        
        N_index = np.where(np.array(structure.atomic_numbers)==N_atom)[0]
        
        all_configurations.extend(build_symmetry_equivalent_configurations(atom_indices,N_index).tolist())
        all_energies.extend([energies_train[i]]*len(build_symmetry_equivalent_configurations(atom_indices,N_index)))
        #print(i,len(all_configurations))
    all_configurations = np.array(all_configurations)
    all_energies = np.array(all_energies)
    
    return all_configurations, all_energies


# In[39]:


def build_ml_qubo(structure,X_train,y_train,max_neigh=1):
    
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    np.fill_diagonal(distance_matrix_filter,[1]*num_sites)
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in X_train:
        matrix = np.outer(config,config)
        upper_tri_elements = matrix[upper_tri_indices]
        descriptor.append(upper_tri_elements)
        

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    
    print('R2: ',reg.score(descriptor, y_train))

    Q = np.zeros((num_sites,num_sites))
    Q[upper_tri_indices] = reg.coef_
    
    return Q
    
#build_ml_qubo(structure,X_train,y_train,max_neigh=1)


# In[38]:


def get_qubo_energies(Q,all_configurations):
    
    predicted_energy = []
    
    for i,config in enumerate(all_configurations):
        predicted_energy.append(classical_energy(config,Q))
    
    return predicted_energy


# In[37]:


def test_qubo_energies(y_pred,y_dft):
    
    from sklearn.metrics import mean_squared_error as mse
    
    return mse(y_pred, y_dft)


# In[36]:


def test_qubo_energies_mape(y_pred,y_dft):
    
    from sklearn.metrics import mean_absolute_percentage_error as mse
    
    return mse(y_pred, y_dft)


# In[35]:


def generate_random_structures(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False):
    
    #N_atoms: number of sites to replace
    #N_config: number of attempts
    #DFT_config: number of final structures generated
    #new_species: new atomic number
    #active_sites: sites in the structure to replace (useful for Al/GaN)
    #atom_indices: indices obtained from get_all_configurations
    #Returns: symmetry independent structures

    all_structures = []

    
    if active_sites is False:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)
        
    

    # Generate a random configurations
    descriptor_all = []
    structures_all = []
    config_all = []
    config_unique = []
    config_unique_count = []
    n_sic = 0
    N_attempts= 0
    
    while n_sic < DFT_config and N_attempts <N_config:
        N_attempts += 1
        sites_index = np.random.choice(num_sites,N_atoms,replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(initial_structure)
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)
#         print(sec[0],np.lexsort(sec,axis=0))
#         print(np.where(np.array(sec)==1))
        # I don't need this if np.unique returns sorted arrays
#         sic = sec[np.lexsort(sec,axis=0)][0]
        sic = sec[0]
        
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)

        if not is_in_config_unique:  
            config_unique.append(sic)

            config_unique_count.append(len(sec))
            n_sic += 1


    final_structures = []

    for config in config_unique:

        N_index = np.where(config==1)[0]
        structure_tmp = copy.deepcopy(initial_structure)
        for N in N_index:
            structure_tmp.replace(N,new_species)
        final_structures.append(structure_tmp)
    if return_multiplicity == True:
        return final_structures,config_unique_count
    else:
        return final_structures



# In[34]:


def generate_random_structures_OLD(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False):
    
    #N_atoms: number of sites to replace
    #N_config: number of attempts
    #DFT_config: number of final structures generated
    #new_species: new atomic number
    #active_sites: sites in the structure to replace (useful for Al/GaN)
    #atom_indices: indices obtained from get_all_configurations
    #Returns: symmetry independent structures

    all_structures = []

    
    if active_sites is False:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)
        
    

    # Generate a random configurations
    descriptor_all = []
    structures_all = []
    config_all = []
    for i in range(N_config):
        sites_index = np.random.choice(num_sites,N_atoms,replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(structure)
        config_all.append(build_symmetry_equivalent_configurations(atom_indices,sites)[0])
    config_unique, config_unique_count = np.unique(config_all,axis=0,return_counts=True)

    final_structures = []
    if len(config_unique) < DFT_config:
        configs = np.arange(len(config_unique))
    else:
        configs = np.random.choice(len(config_unique),DFT_config,replace=False)
    for config in config_unique[configs]:
        #print(config)
        N_index = np.where(config==1)[0]
        structure_tmp = copy.deepcopy(initial_structure)
        for N in N_index:
            structure_tmp.replace(N,new_species)
        final_structures.append(structure_tmp)
    return final_structures

#struct = generate_random_structures(graphene_32,atom_indices=atom_indices,N_atoms=16,new_species=7,N_config=500,DFT_config=10)


# In[33]:


def get_qubo_probability(mu=0.,temperature=298.15, return_all = False):
    graphene_allN_cry_energy_norm_flat = []
    graphene_multiplicity_all_list_flat = []
    graphene_allN_qubo_energy_norm_flat = []
    graphene_allN_qubo_energy_norm_flat_no_pot = []
    num_N = []

    T = temperature
    for i in range(len(graphene_allN_qubo_energy_list)):
        
        graphene_allN_qubo_energy_norm_flat.extend(np.array(graphene_allN_qubo_energy_list[i])+mu*i)
        
        graphene_allN_qubo_energy_norm_flat_no_pot.extend(np.array(graphene_allN_qubo_energy_list[i]))
        graphene_multiplicity_all_list_flat.extend(graphene_multiplicity_all_list[i])

    Z,pi = get_partition_function(graphene_allN_qubo_energy_norm_flat,
                                  graphene_multiplicity_all_list_flat,return_pi=True,T=temperature)
    
   
    
    graphene_allN_qubo_energy_norm_flat = np.round(np.array(graphene_allN_qubo_energy_norm_flat),6)
    graphene_allN_qubo_energy_norm_flat_no_pot = np.round(np.array(graphene_allN_qubo_energy_norm_flat_no_pot),6)
    
    unique_energies, unique_index, unique_counts = np.unique(graphene_allN_qubo_energy_norm_flat_no_pot,
                                                              return_index=True,return_counts=True)
    
    graphene_allN_qubo_energy_norm_flat_no_pot = np.round(graphene_allN_qubo_energy_norm_flat_no_pot,6)
#     print(graphene_allN_qubo_energy_norm_flat_no_pot)
    unique_energies = np.round(unique_energies,6)
    graphene_multiplicity_all_list_flat = np.array(graphene_multiplicity_all_list_flat)
    probability_unique_energies = []
    mult_unique = []
    for ue in unique_energies:
        ue_index = np.where(graphene_allN_qubo_energy_norm_flat_no_pot == ue)[0]

        probability_unique_energies.append(np.sum(pi[ue_index]))
        mult_unique.append(np.sum(graphene_multiplicity_all_list_flat[ue_index]))
    probability_unique_energies = np.array(probability_unique_energies)
    
    if return_all == True:
    
        return unique_energies, probability_unique_energies, Z, mult_unique, uniq
    else:
        return unique_energies, probability_unique_energies
    
# unique_energies,probability_energy, Z, mult = get_qubo_probability_test(mu=0.,temperature=1000, return_all=True)


# In[32]:


def get_qubo_probability_for_T_calc(mu=0.,temperature=298.15, return_all = False):
    graphene_allN_cry_energy_norm_flat = []
    graphene_multiplicity_all_list_flat = []
    graphene_allN_qubo_energy_norm_flat = []
    graphene_allN_qubo_energy_norm_flat_no_pot = []
    num_N = []

    T = temperature
    for i in range(len(graphene_allN_qubo_energy_list)):
        
        graphene_allN_qubo_energy_norm_flat.extend(np.array(graphene_allN_qubo_energy_list[i])+mu*i)
        
        graphene_allN_qubo_energy_norm_flat_no_pot.extend(np.array(graphene_allN_qubo_energy_list[i]))
        graphene_multiplicity_all_list_flat.extend(graphene_multiplicity_all_list[i])

    Z,pi = get_partition_function(graphene_allN_qubo_energy_norm_flat,
                                  graphene_multiplicity_all_list_flat,return_pi=True,T=temperature)
    
   
    
    graphene_allN_qubo_energy_norm_flat = np.round(np.array(graphene_allN_qubo_energy_norm_flat),6)
    graphene_allN_qubo_energy_norm_flat_no_pot = np.round(np.array(graphene_allN_qubo_energy_norm_flat_no_pot),6)
    
    unique_energies, unique_index, unique_counts = np.unique(graphene_allN_qubo_energy_norm_flat,
                                                              return_index=True,return_counts=True)
    
    graphene_allN_qubo_energy_norm_flat = np.round(graphene_allN_qubo_energy_norm_flat,6)
#     print(graphene_allN_qubo_energy_norm_flat_no_pot)
    unique_energies = np.round(unique_energies,6)
    graphene_multiplicity_all_list_flat = np.array(graphene_multiplicity_all_list_flat)
    probability_unique_energies = []
    mult_unique = []
    for ue in unique_energies:
        ue_index = np.where(graphene_allN_qubo_energy_norm_flat == ue)[0]
        probability_unique_energies.append(np.sum(pi[ue_index]))
        mult_unique.append(np.sum(graphene_multiplicity_all_list_flat[ue_index]))
    probability_unique_energies = np.array(probability_unique_energies)
    
    if return_all == True:
    
        return unique_energies, probability_unique_energies, Z, mult_unique
    else:
        return unique_energies, probability_unique_energies
    
# unique_energies,probability_energy, Z, mult = get_qubo_probability_for_T_calc(mu=1.,
#                                               temperature=1000, return_all=True)
# unique_energies


# In[31]:


def get_qubo_probability_old(mu=0.,temperature=298.15):
    graphene_allN_cry_energy_norm_flat = []
    graphene_multiplicity_all_list_flat = []
    graphene_allN_qubo_energy_norm_flat = []
    graphene_allN_qubo_energy_norm_flat_no_pot = []
    num_N = []
#     mu = -0.01
#     temperature = 298.15
    T = temperature
    for i in range(len(graphene_allN_qubo_energy_list)):
        #num_N.extend([i]*len(graphene_allN_qubo_energy_norm[i]))
        #print(np.array(graphene_allN_qubo_energy_norm[i][0]))
        graphene_allN_qubo_energy_norm_flat.extend(np.array(graphene_allN_qubo_energy_list[i])+mu*i)
        
        graphene_allN_qubo_energy_norm_flat_no_pot.extend(np.array(graphene_allN_qubo_energy_list[i]))
        graphene_multiplicity_all_list_flat.extend(graphene_multiplicity_all_list[i])
    #print(graphene_allN_qubo_energy_norm_flat)
    #fig, axs = plt.subplots(1,2,figsize=(15, 5), sharey=False)

    #print(graphene_allN_qubo_energy_norm_flat)
    Z,pi = get_partition_function(graphene_allN_qubo_energy_norm_flat,
                                  graphene_multiplicity_all_list_flat,return_pi=True,T=temperature)

    unique_energies = np.round(np.unique(graphene_allN_qubo_energy_norm_flat_no_pot),6)
    probability_energy = []
    
    for energy in unique_energies:
        index = (np.where(np.round(graphene_allN_qubo_energy_norm_flat_no_pot,6) == energy)[0])
        probability_energy.append(np.sum(pi[index]))
    probability_energy = np.array(probability_energy)
    
    #axs[0].plot(np.arange(len(pi)),pi[sort],'-o',label='QUBO')
    sort = np.argsort(graphene_allN_qubo_energy_norm_flat)[::-1]
    #axs[0].plot(np.arange(len(pi)),np.sort(pi)[::-1],'-',label='QUBO')
    return unique_energies,probability_energy
#get_qubo_probability(mu=a[2],temperature=298.15)


# In[30]:


def get_temperature(unique_energies,probability_energy,Z,mult, return_all = False):
    
    arr = -unique_energies/(k_b*(np.log((probability_energy*Z)/mult)))
    
    if return_all == True:
        return arr
    else:
        
        mask = ~np.isnan(arr)

        # Filter out NaN values using the mask
        arr_without_nan = arr[np.where(arr>0.)[0]]

        return np.average(arr_without_nan)


# In[29]:


def add_num_dopant(dataframe,num_sites,dopant_species):
    config = dataframe.iloc[:,0:num_sites]
    n_dopant = np.sum(config==dopant_species,axis=1)
    dataframe['num_dopants'] = n_dopant
    
    return dataframe.sort_values(by='num_dopants')
#add_num_dopant(ddf,32,7)


# In[21]:


def find_symmetry_equivalent_structures(dataframe, structure, remove_unfeasible = False, species=None ,concentration=None,):
    #spglib-based analysis

    
    #Concentration follows the order given in species
    
    import copy 
    from pymatgen.analysis.structure_matcher import StructureMatcher 

    df = dataframe
    
    num_sites = structure.num_sites
    lattice = structure.lattice
    atom_position = structure.cart_coords
    
    '''if concentration is not None and species is not None:
        feasible_config = []
        all_config = df.iloc[:,0:num_sites].to_numpy()
        #sum_vector = np.sum(all_config,axis=1)
        #feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - vacancies),5))[0]
        for config in all_config:
            feasible = True
            for i in range(len(concentration)):
                feasible *= np.sum(config == species[i]) == concentration[i] 
            #print(feasible)
            feasible_config.append(feasible) 
        
        df = df.iloc[feasible_config,:]'''
    
    if remove_unfeasible == True and species is not None and concentration is not None:
        df = remove_unfeasible_solutions(dataframe,species,concentration)
    
    #configurations = df.iloc[:,0:num_sites].to_numpy()
    
    multiplicity = df['num_occurrences'].to_numpy()
    chain_break = df['chain_break_fraction'].to_numpy()
    energies = df['energy'].to_numpy()


    '''#Replace the C atom with an H atom where the vacancies are
    zero_elements = np.where(configurations == 0) 
    configurations[zero_elements] = 99'''
    
    all_structures = df2structure(df,structure)
    '''for config in configurations:
        all_structures.append(Structure(lattice, config, atom_position, coords_are_cartesian=True))'''

    
    '''#Build the descriptor - WIP
    descriptor = build_descriptor(all_structures)

    descriptor_unique, descriptor_first, descriptor_count = \
    np.unique(descriptor, axis=0,return_counts=True, return_index=True)

    group_structures = []
    for desc in descriptor_unique:
        structure_desc = []
        for i,d in enumerate(descriptor):
            if np.all(np.array(desc) == np.array(d)):
                structure_desc.append(i)
        group_structures.append(structure_desc)'''
    '''for structure in all_structures:
        SpacegroupAnalyzer()
    
    unique_multiplicity = []
    unique_chain_break = []
    unique_structure_index = []
    
    for x in group_structures:
        unique_structure_index.append(x[0])
        unique_multiplicity.append(np.sum(multiplicity[x]))
        unique_chain_break.append(np.average(chain_break[x],weights=multiplicity[x]))    
    
    df = df.iloc[unique_structure_index].copy()
    
    if len(df) == len(unique_multiplicity):
        df['num_occurrences'] = unique_multiplicity
        df['chain_break_fraction'] = unique_chain_break
        
        return df
    
    else:
        print('Some structures might be unfeasible, try using a smaller energy range (lower energy)')
        
        return None'''
    
    #Find the unique structures
    unique_structures = StructureMatcher().group_structures(all_structures)
    
    unique_structures_label = []
    
    #Find to which class the structures belong to
    for structure in all_structures:
        for i in range(len(unique_structures)):
            #print(unique_structures[i][0].composition.reduced_formula,structure.composition.reduced_formula)
            if StructureMatcher().fit(structure,unique_structures[i][0]) == True:
                unique_structures_label.append(i)
                break
    
    unique_structures_label = np.array(unique_structures_label)
    unique_multiplicity = []
    unique_chain_break = []
    for x in range(len(unique_structures)):
        multiplicity_tmp = multiplicity[np.where(unique_structures_label==x)[0]]
        unique_multiplicity.append(np.sum(multiplicity_tmp))
        unique_chain_break.append(np.average(chain_break[np.where(unique_structures_label==x)[0]],weights=multiplicity_tmp))
    
    df = df.iloc[np.unique(unique_structures_label,return_index=True)[1]]
    
    if len(df) == len(unique_multiplicity):
        df1 = df.copy(deep=True)
        df1['num_occurrences'] = unique_multiplicity
        df1['chain_break_fraction'] = unique_chain_break
        
        return df1
    
    else:
        print('Some structures might be unfeasible, try using a smaller energy range (lower energy)')
        
        return None

#ddf = find_symmetry_equivalent_structures(convert_df_binary2atom(df_list[0],[6,7]),graphene_32)


# In[412]:


def get_nodes(problem,embedding):
    embedding = minorminer.find_embedding(problem,qpu_graph)

    data_dict = embedding

    # Create an empty set to store unique values
    unique_values_set = set()

    # Iterate through the values in the dictionary and add them to the set
    for values_list in data_dict.values():
        unique_values_set.update(values_list)

    # Convert the set back to a list to get unique values
    unique_values_list = list(unique_values_set)

    # Sort the list if needed
    unique_values_list.sort()
    
    max_length = 0
    #print(embedding)
    # Iterate through the values in the dictionary and update max_length if needed
    for values_list in data_dict.values():
        current_length = len(values_list)
        if current_length > max_length:
            max_length = current_length

    return len(unique_values_list),max_length


# In[225]:


def build_adjacency_matrix_no_pbc(structure_pbc, max_neigh = 1, diagonal_terms = False, triu = False):
    # structure = pymatgen Structure object
    
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
    num_sites = structure.num_sites
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    distance_matrix = np.zeros((num_sites,num_sites),float)
    
    shells = np.unique(distance_matrix_pbc[0])
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix_pbc == s)[0]
        col_index = np.where(distance_matrix_pbc == s)[1]
        distance_matrix[row_index,col_index] = i
    
    if triu == True:
        distance_matrix = np.triu(distance_matrix,0)
    
    if diagonal_terms == True:
        np.fill_diagonal(distance_matrix,[1]*num_sites)
    
    return distance_matrix


# In[346]:


#ORIGINAL: THIS WORKS, DO NOT MODIFY
def build_ml_h_ryd(structure_pbc,X_train,y_train,max_neigh=1):
    
    C = 5.42e-18#5.42e-24
    Delta_g = 1#-1.25e8
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    #print(distance_matrix)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),float)
    
    ryd_param = [1,1,1/27]
    ryd_param = [1,1,1/27,1/343]
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = ryd_param[i]
    distance_matrix_filter = np.triu(distance_matrix_filter,0)

    #print(distance_matrix_filter[5])    
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0.)
    descriptor = []
    descriptor_test = []
    for config in X_train:
        matrix = np.outer(config,config)*distance_matrix_filter #matrix[i][j] == 1 if i and j are ==1
        upper_tri_elements = matrix[upper_tri_indices]
        
        descriptor.append(upper_tri_elements)
        diag = np.sum(matrix.diagonal())
        diag_all = matrix.diagonal().tolist()
        all_terms = np.sum(upper_tri_elements)
        diag_all.append(all_terms-diag)
        
        descriptor_test.append([diag,all_terms-diag])
        #descriptor_test.append(diag_all)
    #print(descriptor_test)

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    descriptor  = copy.deepcopy(descriptor_test)
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    print(reg.coef_)
    print('R2: ',reg.score(descriptor, y_train))
    
    
    ##########QUBO E
    Q_structure = np.zeros((num_sites,num_sites),float)
    distance_matrix = build_adjacency_matrix(structure,max_neigh=2)
    
    #print(Q_structure)
    nn = np.where(distance_matrix==1)

    Q_structure[nn] = reg.coef_[1]-(reg.coef_[1]*1/27)
    #print(reg.coef_[1],Q_structure[nn])
    nnn = np.where(distance_matrix==2)
    Q_structure[nnn] = reg.coef_[1]*1/27
    #print(Q_structure[nnn])
    # Add the chemical potential (still calling it J1)
    #J1 += J1*mu
    np.fill_diagonal(Q_structure,reg.coef_[0])
    
#     Q = np.zeros((num_sites,num_sites))
#     Q[upper_tri_indices] = reg.coef_
    #print(np.unique(Q_structure))
    return Q_structure




# In[407]:


def build_ml_h_ryd(structure_pbc,X_train,y_train,max_neigh=1):
    
    C = 5.42e-24
    eV_to_rad_s = 1.5193e+15
    Delta_g = 1#-1.25e8
    # Transform the energy from eV to rad/s
    y_train = y_train*eV_to_rad_s
    
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    #print(distance_matrix)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),float)
    
    ryd_param = [1,1,1/27]
    ryd_param = [1,1,1/27,1/343]
    
    #ryd_param = [1,C,C/27,C/343]
    #ryd_param = [1,(1/C)**(1/6),(1.73/C)**(1/6),(2/C)**(1/6)]
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = ryd_param[i]
    distance_matrix_filter = np.triu(distance_matrix_filter,0)

    #print(distance_matrix_filter[5])    
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0.)
    descriptor = []
    descriptor_test = []
    for config in X_train:
        matrix = np.outer(config,config)*distance_matrix_filter #matrix[i][j] == 1 if i and j are ==1
        upper_tri_elements = matrix[upper_tri_indices]
        
        descriptor.append(upper_tri_elements)
        diag = np.sum(matrix.diagonal())
        diag_all = matrix.diagonal().tolist()
        all_terms = np.sum(upper_tri_elements)
        diag_all.append(all_terms-diag)
        
        descriptor_test.append([diag,all_terms-diag])
        #descriptor_test.append(diag_all)
    #print(descriptor_test)

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    descriptor  = copy.deepcopy(descriptor_test)
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    print(reg.coef_)
    print((C/reg.coef_[1])**(1/6))
    print('R2: ',reg.score(descriptor, y_train))
    
    
    ##########QUBO E
    Q_structure = np.zeros((num_sites,num_sites),float)
    distance_matrix = build_adjacency_matrix(structure,max_neigh=2)
    
    #print(Q_structure)
    nn = np.where(distance_matrix==1)

    Q_structure[nn] = reg.coef_[1]-(reg.coef_[1]*1/27)
    #print(reg.coef_[1],Q_structure[nn])
    nnn = np.where(distance_matrix==2)
    Q_structure[nnn] = reg.coef_[1]*1/27
    #print(Q_structure[nnn])
    # Add the chemical potential (still calling it J1)
    #J1 += J1*mu
    np.fill_diagonal(Q_structure,reg.coef_[0])
    
#     Q = np.zeros((num_sites,num_sites))
#     Q[upper_tri_indices] = reg.coef_
    #print(np.unique(Q_structure))
    return Q_structure




# In[999]:


def build_ml_qubo(structure,X_train,y_train,max_neigh=1):
    
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    np.fill_diagonal(distance_matrix_filter,[1]*num_sites)
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in X_train:
        matrix = np.outer(config,config)
        upper_tri_elements = matrix[upper_tri_indices]
        descriptor.append(upper_tri_elements)

    
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    
    print('R2: ',reg.score(descriptor, y_train))

    Q = np.zeros((num_sites,num_sites))
    Q[upper_tri_indices] = reg.coef_
    
    return Q
    


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]




def get_classical_av_conc(Q_gaaln_ml,mu_range,T,size=10000):


    size = size
    binary_vector = []
    QUBO_classical_E = []
    concentration = []

    # Generate binary vectors and compute classical energy
    for conc in np.random.randint(0, 54, size=size):
        concentration.append(conc)
        ones = np.random.choice(54, conc, replace=False)
        x = np.zeros(54, dtype='int')
        x[ones] = 1
        binary_vector.append(x)
        QUBO_classical_E.append(classical_energy(x, Q_gaaln_ml))

    # Convert lists to numpy arrays
    binary_vector = np.array(binary_vector)
    QUBO_classical_E = np.array(QUBO_classical_E)
    concentration = np.array(concentration)

    av_conc_classical = []
    mu_range = np.array(mu_range)
    mu_all = mu_range*Q_gaaln_ml[0][0]
    for mu in mu_all:
        #print(mu)
        for i in range(len(QUBO_classical_E)):
            #print(mu)
            energy_new = QUBO_classical_E + concentration*mu
        Z,pi = get_partition_function(energy_new,[1]*len(QUBO_classical_E),return_pi=True,T=T)
        av_conc_classical.append(np.sum(pi*concentration))
    av_conc_classical = np.array(av_conc_classical)/54
    
    return av_conc_classical



def generate_random_structures(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False):
    
    #N_atoms: number of sites to replace
    #N_config: number of attempts
    #DFT_config: number of final structures generated
    #new_species: new atomic number
    #active_sites: sites in the structure to replace (useful for Al/GaN)
    #atom_indices: indices obtained from get_all_configurations
    #Returns: symmetry independent structures

    all_structures = []

    
    if active_sites is False:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)
        
    

    # Generate a random configurations
    descriptor_all = []
    structures_all = []
    config_all = []
    config_unique = []
    config_unique_count = []
    n_sic = 0
    N_attempts= 0
    
    while n_sic < DFT_config and N_attempts <N_config:
        N_attempts += 1
        sites_index = np.random.choice(num_sites,N_atoms,replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(structure)
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)
#         print(sec[0],np.lexsort(sec,axis=0))
#         print(np.where(np.array(sec)==1))
        # I don't need this if np.unique returns sorted arrays
#         sic = sec[np.lexsort(sec,axis=0)][0]
        sic = sec[0]
        
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)

        if not is_in_config_unique:  
            config_unique.append(sic)

            config_unique_count.append(len(sec))
            n_sic += 1


    final_structures = []

    for config in config_unique:

        N_index = np.where(config==1)[0]
        structure_tmp = copy.deepcopy(initial_structure)
        for N in N_index:
            structure_tmp.replace(N,new_species)
        final_structures.append(structure_tmp)
    if return_multiplicity == True:
        return final_structures,config_unique_count
    else:
        return final_structures



