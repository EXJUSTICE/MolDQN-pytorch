"""
Hyperparameters for Constrained-Objective Optimization.
"""

similarity_weight=0.2,
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000
optimizer = "Adam"
polyak = 0.995
atom_types = ["C", "O", "N"]
init_mol = None
max_steps_per_episode = 40
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]
replay_buffer_size = 1000000
learning_rate = 1e-4
gamma = 0.95
fingerprint_radius = 3
fingerprint_length = 2048
discount_factor = 0.9
