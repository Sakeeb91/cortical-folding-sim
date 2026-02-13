"""Forward simulation demo: sphere → folded cortex."""

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cortical_folding.mesh import build_topology
from cortical_folding.synthetic import (
    create_anisotropy_field,
    create_icosphere,
    create_skull,
    create_regional_growth,
)
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.losses import gyrification_index
from cortical_folding.viz import plot_mesh, plot_growth_field, plot_simulation_frames


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anisotropic",
        action="store_true",
        help="Enable anisotropic directional growth redistribution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Creating icosphere mesh...")
    vertices, faces = create_icosphere(subdivisions=3, radius=1.0)
    topo = build_topology(vertices, faces)
    print(f"  Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}, Edges: {topo.edges.shape[0]}")

    skull_center, skull_radius = create_skull(radius=1.5)

    # Regional growth: high on top hemisphere
    growth_rates = create_regional_growth(
        vertices, topo.faces, high_rate=0.8, low_rate=0.1, axis=2, threshold=0.0
    )
    face_anisotropy = create_anisotropy_field(
        mode="regional" if args.anisotropic else "none",
        vertices=vertices,
        faces=topo.faces,
        high_value=1.0,
        low_value=0.0,
        axis=2,
        threshold=0.0,
    )
    print(f"  Growth rates: min={float(growth_rates.min()):.2f}, max={float(growth_rates.max()):.2f}")

    params = SimParams(
        Kc=2.0,
        Kb=3.0,
        damping=0.9,
        skull_center=skull_center,
        skull_radius=skull_radius,
        skull_stiffness=100.0,
        carrying_cap_factor=4.0,
        tau=500.0,
        dt=0.02,
        anisotropy_strength=0.35 if args.anisotropic else 0.0,
        anisotropy_axis=jnp.array([0.0, 0.0, 1.0]),
    )

    initial_state = make_initial_state(vertices, topo)
    print(f"\nRunning forward simulation (200 steps)...")
    final_state, trajectory = simulate(
        initial_state, topo, growth_rates, params, face_anisotropy=face_anisotropy, n_steps=200
    )

    gi = gyrification_index(final_state.vertices, topo, skull_radius)
    print(f"  Final gyrification index: {float(gi):.3f}")

    # Visualize
    fig = plot_simulation_frames(
        trajectory, topo.faces,
        steps=[0, 50, 100, 150, 199],
        title="Cortical Folding: Sphere → Folded Surface",
    )
    plt.savefig("forward_simulation.png", dpi=150, bbox_inches="tight")
    print("  Saved forward_simulation.png")

    fig2 = plot_growth_field(vertices, topo.faces, growth_rates, title="Input Growth Field")
    plt.savefig("growth_field.png", dpi=150, bbox_inches="tight")
    print("  Saved growth_field.png")

    plt.show()


if __name__ == "__main__":
    main()
