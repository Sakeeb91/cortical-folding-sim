"""Generate an animated cortical folding simulation (.gif or .mp4)."""

from __future__ import annotations

import argparse

import jax.numpy as jnp

from cortical_folding.mesh import build_topology
from cortical_folding.synthetic import create_icosphere, create_regional_growth, create_skull
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.viz import save_simulation_animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="forward_simulation.gif",
        help="Animation output path (.gif or .mp4).",
    )
    parser.add_argument("--subdivisions", type=int, default=3, help="Icosphere subdivisions.")
    parser.add_argument("--steps", type=int, default=220, help="Number of simulation steps.")
    parser.add_argument("--fps", type=int, default=20, help="Animation frame rate.")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for animation.")
    parser.add_argument("--rotate", action="store_true", help="Rotate camera over time.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Creating mesh and growth field...")
    vertices, faces = create_icosphere(subdivisions=args.subdivisions, radius=1.0)
    topo = build_topology(vertices, faces)
    skull_center, skull_radius = create_skull(radius=1.5)
    growth_rates = create_regional_growth(
        vertices, topo.faces, high_rate=0.8, low_rate=0.1, axis=2, threshold=0.0
    )

    # Robust defaults intended for long trajectories.
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
        max_growth_rate=1.2,
        max_force_norm=400.0,
        max_acc_norm=300.0,
        max_velocity_norm=1.2,
        max_displacement_per_step=0.03,
        enable_self_collision=True,
        self_collision_min_dist=0.03,
        self_collision_stiffness=40.0,
        self_collision_n_sample=512,
    )

    print(f"Running simulation ({args.steps} steps)...")
    state = make_initial_state(vertices, topo)
    _, trajectory = simulate(state, topo, growth_rates, params, n_steps=args.steps)

    print(f"Saving animation to {args.output}...")
    save_simulation_animation(
        trajectory,
        topo.faces,
        output_path=args.output,
        fps=args.fps,
        stride=args.stride,
        rotate=args.rotate,
    )
    print("Done.")


if __name__ == "__main__":
    main()
