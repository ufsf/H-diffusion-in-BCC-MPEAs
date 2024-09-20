import numpy as np

def unwrap_coord(H_diff, Box):
    def process_component(Hu):
        # Calculate differences between consecutive elements
        dHu = np.diff(Hu)
        dHu_adjusted = dHu.copy()

        # Maximum and minimum differences
        dHu_max = np.max(dHu)
        dHu_min = np.min(dHu)

        # Adjust differences exceeding half the box size
        dHu_adjusted[dHu > dHu_max - Box / 2] -= Box
        dHu_adjusted[dHu < dHu_min + Box / 2] += Box

        # Unwrap by summing adjusted differences starting from the first element
        Hu_unwarp = np.zeros_like(Hu)
        Hu_unwarp[0] = Hu[0]
        Hu_unwarp[1:] = Hu_unwarp[0] + np.cumsum(dHu_adjusted)
        return Hu_unwarp

    # Process each coordinate component
    Hx_unwarp = process_component(H_diff[:, 0])
    Hy_unwarp = process_component(H_diff[:, 1])
    Hz_unwarp = process_component(H_diff[:, 2])

    return Hx_unwarp, Hy_unwarp, Hz_unwarp
