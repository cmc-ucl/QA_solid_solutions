
import numpy as np
import copy

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from CRYSTALpytools.convert import cry_gui2pmg

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



