"""Quick demo: sphere → folded cortex in ~20 lines."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from cortical_folding.mesh import build_topology
from cortical_folding.synthetic import create_icosphere, create_uniform_growth
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.viz import plot_simulation_frames

vertices, faces = create_icosphere(subdivisions=3, radius=1.0)
topo = build_topology(vertices, faces)
growth = create_uniform_growth(topo.faces.shape[0], rate=0.5)
params = SimParams(
    Kc=2.0, Kb=3.0, damping=0.9,
    skull_center=jnp.zeros(3), skull_radius=1.5,
    skull_stiffness=100.0, carrying_cap_factor=4.0, tau=500.0, dt=0.02,
)
state = make_initial_state(vertices, topo)
final, traj = simulate(state, topo, growth, params, n_steps=200)
plot_simulation_frames(traj, topo.faces, steps=[0, 50, 100, 150, 199])
plt.savefig("demo_folding.png", dpi=150, bbox_inches="tight")
print(f"Done! Vertices: {vertices.shape[0]}, saved demo_folding.png")
plt.show()
