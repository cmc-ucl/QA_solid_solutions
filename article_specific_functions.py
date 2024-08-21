import numpy as np
from QA_solid_solutions_functions import *

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


