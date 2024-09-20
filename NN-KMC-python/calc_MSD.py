import numpy as np
from unwrap_coord import unwrap_coord

def calc_MSD(H_diff, Box, Len, Time):
    Hx_unwarp, Hy_unwarp, Hz_unwarp = unwrap_coord(H_diff, Box)
    H_unwarp = np.column_stack((Hx_unwarp, Hy_unwarp, Hz_unwarp))

    if len(Hx_unwarp) != len(Time):
        raise ValueError('Mismatch between unwrapped coordinates and time data lengths.')

    np.random.seed()  # for reproducibility, consider setting a specific seed or removing this line
    start_point = np.random.randint(len(Hx_unwarp) // 10, len(Hx_unwarp) * 9 // 10)

    # Split the unwrapped coordinates at the start point
    first_half = (Hx_unwarp[:start_point], Hy_unwarp[:start_point], Hz_unwarp[:start_point], Time[:start_point])
    second_half = (Hx_unwarp[start_point:], Hy_unwarp[start_point:], Hz_unwarp[start_point:], Time[start_point:])

    def extract_significant_movements(hx, hy, hz, time):
        extracted = []
        x_current, y_current, z_current = hx[0], hy[0], hz[0]
        for x, y, z, t in zip(hx, hy, hz, time):
            distance = np.sqrt((x - x_current) ** 2 + (y - y_current) ** 2 + (z - z_current) ** 2)
            if distance > Len:
                extracted.append((x, y, z, t))
                x_current, y_current, z_current = x, y, z
        return np.array(extracted)

    # Extract significant movements from both halves
    movements_first = extract_significant_movements(*first_half)
    movements_second = extract_significant_movements(*second_half)
    all_movements = np.vstack([movements_first, movements_second]) if movements_first.size and movements_second.size else np.array([]).reshape(0,4)

    # Calculate MSD if there are enough points
    if all_movements.size > 0:
        dx = np.diff(all_movements[:, 0])
        dy = np.diff(all_movements[:, 1])
        dz = np.diff(all_movements[:, 2])
        MSD = np.cumsum(dx**2 + dy**2 + dz**2)
        coord_final = {'x': all_movements[:, 0], 'y': all_movements[:, 1], 'z': all_movements[:, 2]}
        Time_extract_end = all_movements[-1, 3]
    else:
        MSD = np.array([])
        coord_final = {'x': [], 'y': [], 'z': []}
        Time_extract_end = np.nan

    return MSD, H_unwarp, coord_final, Time_extract_end
