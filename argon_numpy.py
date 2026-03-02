import numpy as np
import time
epsilon = 121.0
sigma = 3.4

def initialize_cluster(num_atoms):
    """
    Create a random cluster of atoms in 3D space.
    Returns a numpy array of shape (N,3)
    """
    return np.random.uniform(-1, 1, size=(num_atoms, 3))


def translational_move(positions, temp, stepsize):
    N = positions.shape[0]

    for i in range(N):
        move = (np.random.rand(3) - 0.5) * 2 * stepsize
        new_pos = positions.copy()
        new_pos[i] += move

        old_energy = single_atom_energy(positions,i)
        new_energy = single_atom_energy(new_pos,i)
        delta = new_energy - old_energy

        if delta < 0 or np.random.rand() < np.exp(-delta/temp):
            positions[i] += move  # update in place

    return positions


def COM(positions):
    """
    Center the cluster at the origin.
    positions: np.array (N,3)
    """
    mean = positions.mean(axis=0)
    positions -= mean
    return positions


def calc_energy_vec(positions):
    """
    Calculate total Lennard-Jones energy vectorized.
    positions: np.array shape (N,3)
    """
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    i_upper = np.triu_indices(len(positions), k=1)
    r = distances[i_upper]
    sr6 = (sigma / r)**6
    return np.sum(4 * epsilon * (sr6**2 - sr6))


def graph_coords(positions):
    """
    Returns x, y, z for plotting.
    """
    return positions[:,0], positions[:,1], positions[:,2]

def single_atom_energy(positions, atom_index, epsilon=121.0, sigma=3.4):
    """
    Calculate the Lennard-Jones energy contribution of one atom
    relative to all others.
    positions: np.array of shape (N,3)
    atom_index: int, which atom to calculate
    """
    # all positions except the atom itself
    others = np.delete(positions, atom_index, axis=0)
    atom_pos = positions[atom_index]

    # vectorized distances
    diffs = others - atom_pos  # shape (N-1, 3)
    r2 = np.sum(diffs**2, axis=1)

    sr6 = sigma**6/(r2**3)
    energy = np.sum(4 * epsilon * (sr6**2 - sr6))
    return energy

def quench(initial_t):
    return initial_t * .99 #not sure if this is a good quench

'''
before = time.time()

test = initialize_cluster(13)
steps = 100000
initial_t = 10
for i in range(steps):
    translational_move(test,quench(initial_t, steps, i),.125)
    COM(test)

print(calc_energy_vec(test))

print(time.time()-before)
'''