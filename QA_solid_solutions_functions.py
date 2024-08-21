import numpy as np
import copy
import re

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from CRYSTALpytools.convert import cry_gui2pmg
from scipy import constants
k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]

from dwave.embedding.chain_strength import  uniform_torque_compensation
from dwave.system import EmbeddingComposite, DWaveSampler
import minorminer
import dimod 

def add_num_dopant(dataframe,num_sites,dopant_species):
    config = dataframe.iloc[:,0:num_sites]
    n_dopant = np.sum(config==dopant_species,axis=1)
    dataframe['num_dopants'] = n_dopant
    
    return dataframe.sort_values(by='num_dopants')

def binomial_coefficient(n, k):
    return np.factorial(n) // (np.factorial(k) * np.factorial(n - k))

def adjacency_matrix_no_pbc(structure_pbc, max_neigh = 1, diagonal_terms = False, triu = False):
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

def build_binary_vector(atomic_numbers,atom_types=None):
    """Summary line.

    Extended description of function.

    Args:
        atomic_numbers (list): List of atom number of the sites in the structure
        atom_types (list): List of 2 elements. List element 0 = atomic_number of site == 0, 
                           list element 1 = atomic_number of site == 1

    Returns:
        List: Binary list of atomic numbers

    """
    
    atomic_numbers = np.array(atomic_numbers)
    num_sites = len(atomic_numbers)
    
    if atom_types == None:
        species = np.unique(atomic_numbers)
    else:
        species = atom_types
    
    binary_atomic_numbers = np.zeros(num_sites,dtype=int)
    
    for i,species_type in enumerate(species):
        #print(i,species_type)
        sites = np.where(atomic_numbers == species_type)[0]
        #print(i,species_type,sites)
        binary_atomic_numbers[sites] = i
    
    return binary_atomic_numbers

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

def classical_energy(x,q):
    # x is the binary vector
    # q is the qubo matrix

    E_tmp = np.matmul(x,q)
    E_classical = np.sum(x*E_tmp)
    
    return E_classical

def extract_elastic_moduli(lines):
    # Find the start of the elastic moduli table
    start_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'TOTAL ELASTIC MODULI \(kBar\)', line):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError("Elastic moduli section not found in the lines.")
    
    # Skip the first two lines (header and separator)
    data_lines = lines[start_idx+3:]
    
    matrix = []
    for line in data_lines:
        if re.match(r'\s*[-]+', line):  # Stop at the ending separator
            break
        # Split each line by spaces and filter out the first element (the direction label)
        values = list(map(float, line.split()[1:]))
        matrix.append(values)
    
    # Convert to a NumPy array
    return np.array(matrix)

#MAKE THIS GENERAL
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

def get_nodes(problem,embedding):

    qpu_sampler = DWaveSampler(solver=dict(topology__type='pegasus'))
    qpu_graph = qpu_sampler.to_networkx_graph() 

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
    
    energy = np.array(energy)
    multiplicity = np.array(multiplicity)
    p_i = multiplicity * np.exp((-energy + (N_N * N_potential)) / (k_b * T))
    pf = np.sum(p_i)
    
    p_i /= pf
    
    if return_pi:
        return pf, p_i
    else:
        return pf
    
def get_qubo_energies(Q,all_configurations):
    
    predicted_energy = []
    
    for i,config in enumerate(all_configurations):
        predicted_energy.append(classical_energy(config,Q))
    
    return predicted_energy

def get_temperature(unique_energies,probability_energy,Z,mult, return_all = False):
    
    arr = -unique_energies/(k_b*(np.log((probability_energy*Z)/mult)))
    
    if return_all == True:
        return arr
    else:
        
        mask = ~np.isnan(arr)

        # Filter out NaN values using the mask
        arr_without_nan = arr[np.where(arr>0.)[0]]

        return np.average(arr_without_nan)

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def test_qubo_energies(y_pred,y_dft):
    
    from sklearn.metrics import mean_squared_error as mse
    
    return mse(y_pred, y_dft)


def test_qubo_energies_mape(y_pred,y_dft):
    
    from sklearn.metrics import mean_absolute_percentage_error as mse
    
    return mse(y_pred, y_dft)