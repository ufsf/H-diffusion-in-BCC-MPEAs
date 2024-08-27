import numpy as np
from ase.build import bulk
from ase.io import write
import numpy as np
import os
from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
import torch
import torch.nn as nn

input_size = 3696  # Example input size
hidden_layers = [32, 32, 32, 32]  # List of hidden layer sizes
output_size = 1  # Single output for regression
batch_size = 128
learning_rate = 0.001
num_epochs = 100
patience = 20

class SOAPNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(SOAPNet, self).__init__()
        layers = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
model = SOAPNet(input_size, hidden_layers, output_size)
model.load_state_dict(torch.load('../../NN-SE/best_model_32_32_32_32.pth'))

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
        x = x - xlo
        y = y - ylo
        z = z - zlo
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

### Load all T sites and T-T paths
Size = 20
Lattice = 3.2
H_coords = np.loadtxt('../../H-Path/path'+str(Size)+'/H_coords.txt')
cell0 = 3.2*Size
All_TT = np.loadtxt('../../H-Path/path'+str(Size)+'/All_TT.txt', dtype=int);

### Deine SOAP
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
cutoff_distance = 6

HEA_ideal = read_lammps_data('ideal.data')
HEA_relaxed = read_lammps_data('relaxed.data')

model.eval()  # Set the model to evaluation mode

MAX_BATCH_SIZE = 40000  # Define a sensible batch size based on your hardware capability

curr_box = HEA_ideal.get_cell()[0, 0]
ratio = curr_box/cell0
curr_H_coords = H_coords*ratio
displacements = HEA_relaxed.get_positions() - HEA_ideal.get_positions()

SSE_all = []
batch_inputs = []
for index, (initial, final) in enumerate(All_TT):
    H_ini = curr_H_coords[initial - 1]
    T_distorted, dist_array, weights = calculate_distortion(HEA_ideal, H_ini, cutoff_distance, displacements)
    T_pred = H_ini + T_distorted
    T_pred = T_pred.reshape(1, -1)
    T_soap_descriptor = soap.create(HEA_relaxed, T_pred)
    T_soap_vector = np.mean(T_soap_descriptor, axis=0)

    H_fi = curr_H_coords[final - 1]
    
    delta_H_positions, distance = find_mic(H_fi - H_ini, HEA_ideal.cell, pbc=True)
    avg_H_positions = H_ini + delta_H_positions / 2
    
    S_distorted, dist_array, weights = calculate_distortion(HEA_ideal, avg_H_positions, cutoff_distance, displacements)
    S_pred = avg_H_positions + S_distorted
    S_pred = S_pred.reshape(1, -1)
    
    
    S_soap_descriptor = soap.create(HEA_relaxed, S_pred)
    S_soap_vector = np.mean(S_soap_descriptor, axis=0)  # Averaging if there are multiple H atoms or features
    
    soap_vector = S_soap_vector - T_soap_vector

    batch_inputs.append(soap_vector)
    
    # Check if we reached the maximum batch size or end of list
    if len(batch_inputs) >= MAX_BATCH_SIZE or index == len(All_TT) - 1:
        # Convert list of inputs into a tensor
        # input_tensor = torch.tensor(batch_inputs).float()

        batch_inputs_array = np.array(batch_inputs)  # Convert list of arrays to a single NumPy array
        input_tensor = torch.tensor(batch_inputs_array).float()  # Convert the NumPy array to a tensor

        # Perform batch inference
        with torch.no_grad():
            outputs = model(input_tensor)
            SSE_all.extend(outputs.numpy().flatten().tolist())
            outputs = torch.exp(outputs)  # Ensure outputs are positive

        # Reset batch inputs after processing
        batch_inputs = []


print(f'barrier completed')
np.save('Barrier.npy', SSE_all)