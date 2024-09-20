import numpy as np
from scipy.io import savemat
from unwrap_coord import unwrap_coord
from calc_MSD import calc_MSD

def CalcD(imat, Temp, MSD_max, N):
    # Constants
    kb = 8.617333262e-5  # Boltzmann constant in eV/K
    v0 = 1e13  # Attempt frequency (1/s)

    # Load simulation parameters
    Len = np.loadtxt(f'len{N}.txt')
    Box = Len[imat]

    # Load path and barrier information
    All_path_id = np.loadtxt(f'../H-Path/path{N}/All_TT.txt')
    num_site = len(np.unique(All_path_id[:, 0]))

    # Load and scale hydrogen coordinates
    H_coords = np.loadtxt(f'../H-Path/path{N}/H_coords.txt') * (Box / (3.2 * N))

    # Load energy barriers
    Barrier = np.exp(np.load(f'../All-Barrier-{N}/{imat}.npy'))
    SE = np.load(f'../All-Barrier-{N}/{imat}-TSE.npy')

    # Initialize simulation
    curr_id = np.random.choice(num_site, 1)[0]
    Time = 0.0
    H_wrap = [H_coords[curr_id, :]]
    Time_store = [0]
    SE_store = [SE[curr_id]]
    DB_store = []

    count = 0
    MSD_tmp = []

    # Run KMC simulation
    while True:
        next_indices = np.where(All_path_id[:, 0] == curr_id)[0]
        next_sites = All_path_id[next_indices, 1]
        barriers = Barrier[next_indices]
        rates = v0 * np.exp(-barriers / (kb * Temp))
        total_rate = np.sum(rates)
        time_increment = -np.log(np.random.random()) / total_rate
        Time += time_increment

        # Select next site based on rates
        probabilities = rates / total_rate
        next_index = np.random.choice(next_indices, p=probabilities)
        curr_id = int(next_sites[next_index])
        DB_store.append(barriers[next_index])

        H_wrap.append(H_coords[curr_id, :])
        Time_store.append(Time)
        SE_store.append(SE[curr_id])

        count += 1

        # Compute MSD every 10000 steps
        if count % 10000 == 0:
            MSD_result, _, _, _ = calc_MSD(np.array(H_wrap), Box, 10, np.array(Time_store))
            MSD_tmp = MSD_result
            if MSD_tmp and MSD_tmp[-1] > MSD_max:
                break

        # Restart KMC if stuck
        if count > 100000 and MSD_tmp and MSD_tmp[-1] < 100:
            curr_id = np.random.choice(num_site, 1)[0]
            Time = 0
            H_wrap = [H_coords[curr_id, :]]
            Time_store = [0]
            SE_store = [SE[curr_id]]
            DB_store = []
            count = 1
            print('Restart KMC due to lack of progress.')

    # Post-process results
    MSD, H_unwrap, coord, Time_final = calc_MSD(np.array(H_wrap), Box, 10, np.array(Time_store))

    res = {
        'D_all': MSD[-1] / 6 / Time_final[-1] * 1e-20,
        'MSD': MSD,
        'H_wrap': np.array(H_wrap),
        'H_unwrap': H_unwrap,
        'H_unwrap_extract': coord,
        'Time_f': Time_final,
        'DB_store': DB_store,
        'SE_store': SE_store
    }
    return res

# Simulation parameters
imat = 1
temp = np.arange(300, 2100, 100)  # Temperature range
N = 20
MSD_max = 1e5

# Run simulation for each temperature
for T in temp:
    result = CalcD(imat, T, MSD_max, N)
    filename = f"{T}-{imat}.mat"
    savemat(filename, {'res': result})
    print(f"Results saved for temperature {T} K.")
