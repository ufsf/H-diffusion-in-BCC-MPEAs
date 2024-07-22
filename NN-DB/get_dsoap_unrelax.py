import numpy as np
import os
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP

def read_lammps_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read box dimensions
    xlo, xhi = [float(i) for i in lines[5].split()[:2]]
    ylo, yhi = [float(i) for i in lines[6].split()[:2]]
    zlo, zhi = [float(i) for i in lines[7].split()[:2]]
    cell = [[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]]

    # Read atom information
    atom_lines = lines[12:]
    atom_data = []
    for line in atom_lines:
        if line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atom_data.append((atom_id, atom_type, x, y, z))

    # Sort atoms by atom_id
    atom_data.sort(key=lambda atom: atom[0])

    # Map atom types to elements (excluding hydrogen)
    element_map = {1: "Mo", 2: "Nb", 3: "Ta", 4: "W"}
    symbols = [element_map[atom[1]] for atom in atom_data]
    positions = np.array([[atom[2], atom[3], atom[4]] for atom in atom_data])

    # Create ASE Atoms object without hydrogen atoms
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    return atoms

def read_lammps_data_unrelax(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the box dimensions directly using the known line numbers for dimensions
    xlo, xhi = [float(i) for i in lines[3].split()[:2]]
    ylo, yhi = [float(i) for i in lines[4].split()[:2]]
    zlo, zhi = [float(i) for i in lines[5].split()[:2]]
    cell = [[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]]

    # Start reading atom data after the line 'Atoms # atomic'
    atom_lines = lines[7:]  # Correct the index based on your file's structure
    # print(atom_lines)
    atom_data = []
    for line in atom_lines:
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atom_data.append((atom_id, atom_type, x, y, z))

    # Map atom types to elements
    element_map = {1: "Mo", 2: "Nb", 3: "Ta", 4: "W"}
    symbols = [element_map[atom[1]] for atom in atom_data if atom[1] in element_map]
    positions = np.array([[atom[2], atom[3], atom[4]] for atom in atom_data])
    # print(positions)
    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def calculate_distortion(atoms, position, cutoff, displacements):
    # Ensure the Atoms object has proper periodic boundary conditions and cell set
    if not atoms.pbc.any():
        raise ValueError("Periodic boundary conditions not set for Atoms object.")
    
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # Calculate vector distances considering PBC
    # Calculate the difference vector from position to all atoms' positions
    deltas = positions - position

    # Apply minimum image convention using the cell
    deltas -= np.round(deltas / cell.lengths()) * cell.lengths()

    # Calculate scalar distances from these corrected vectors
    distances = np.linalg.norm(deltas, axis=1)

    # Filter and collect neighbors within the cutoff distance
    neighbors = [(idx, dist) for idx, dist in enumerate(distances) if dist < cutoff]

    # Sort the list of tuples (index, distance) by distance
    neighbors.sort(key=lambda x: x[1])

    displacement_tmp = []
    dist_tmp = []
    for idx, dist in neighbors:
        displacement_tmp.append(displacements[idx])
        dist_tmp.append(dist)

    dist_array = np.array(dist_tmp)
    sigma = 0.1
    weights = np.exp(-(dist_array**2) / (2 * sigma**2))
    weights = weights[:, np.newaxis]
    weights = weights/weights.sum(axis=0)

    displacement_array = np.array(displacement_tmp)

    weighted_displacement = displacement_array * weights
    T_distorted = weighted_displacement.sum(axis=0)
    return T_distorted, dist_array, weights


H_coords = np.loadtxt('H_coords.txt')
cell0 = np.loadtxt('cell.txt');
cell0 = cell0[0]
Unique_TT = np.loadtxt('Unique-TT.txt', dtype=int);


# distance_all = []
species = ["Mo", "Nb", "Ta", "W"]
r_cut = 7.0
n_max = 8
l_max = 6

soap = SOAP(
    species=species,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    periodic=True
)
    

cutoff_distance = 6.0  # Define your cutoff distance in Ångstroms
dataset = 'all'
for i in range(969):
    print(i)
    relax = os.path.join(dataset, f"{i}-relax.data")
    atom_relax = read_lammps_data(relax)
    curr_box = atom_relax.get_cell()[0,0]
    ratio = curr_box/cell0
    curr_H_coords = H_coords*ratio
    unrelax = os.path.join(dataset, f"{i}.data")
    atom_unrelax = read_lammps_data_unrelax(unrelax)

    # Calculate displacements
    displacements = atom_relax.get_positions() - atom_unrelax.get_positions()
    # atom_unrelax_position = atom_unrelax.get_positions()
    # all_soap_T = []
    # for iT in range(1,769):
    #     H_iT = curr_H_coords[iT-1]
    #     T_distorted, dist_array, weights = calculate_distortion(atom_unrelax, H_iT, cutoff_distance, displacements)
    #     T_pred = H_iT + T_distorted
    #     T_pred = T_pred.reshape(1, -1)
    #     soap_descriptor = soap.create(atom_relax,T_pred)
    #     soap_vector = np.mean(soap_descriptor, axis=0)  # Averaging if there are multiple H atoms or features
    #     all_soap_T.append(soap_vector)

    # np.save('soap_T_'+str(i)+'.npy', all_soap_T)

    d_soap_all = []
    for initial, final in Unique_TT:
        H_ini = curr_H_coords[initial-1]
        T_distorted, dist_array, weights = calculate_distortion(atom_unrelax, H_ini, cutoff_distance, displacements)
        T_ini_pred = H_ini + T_distorted
        T_ini_pred = T_ini_pred.reshape(1, -1)
        H_ini = H_ini.reshape(1, -1)
        T_soap_descriptor = soap.create(atom_unrelax,H_ini)
        T_soap_vector = np.mean(T_soap_descriptor, axis=0)  # Averaging if there are multiple H atoms or features

        H_fi = curr_H_coords[final-1]
        delta_H_positions, distance = find_mic(H_fi - H_ini, atom_unrelax.cell, pbc=True)
        avg_H_positions = H_ini + delta_H_positions/2
        S_distorted, dist_array, weights = calculate_distortion(atom_unrelax, avg_H_positions, cutoff_distance, displacements)
        S_pred = avg_H_positions + S_distorted
        S_pred = S_pred.reshape(1, -1)
        avg_H_positions = avg_H_positions.reshape(1, -1)
        S_soap_descriptor = soap.create(atom_unrelax,avg_H_positions)
        S_soap_vector = np.mean(S_soap_descriptor, axis=0)  # Averaging if there are multiple H atoms or features

        dsoap_vector = S_soap_vector - T_soap_vector

        d_soap_all.append(dsoap_vector)
    np.save('dsoap_'+str(i)+'.npy', d_soap_all)

