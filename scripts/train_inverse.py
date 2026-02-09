"""Inverse problem: recover growth field from observed folded surface."""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt

from cortical_folding.mesh import build_topology
from cortical_folding.synthetic import (
    create_icosphere,
    create_skull,
    create_regional_growth,
    create_target_folded,
)
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.losses import total_loss, LossWeights, gyrification_index
from cortical_folding.growth_net import GrowthFieldNet, extract_vertex_features, growth_rates_to_faces
from cortical_folding.viz import plot_growth_field, plot_mesh


def main():
    key = jax.random.PRNGKey(42)

    print("Setting up synthetic target...")
    vertices, faces = create_icosphere(subdivisions=3, radius=1.0)
    topo = build_topology(vertices, faces)
    skull_center, skull_radius = create_skull(radius=1.5)

    # Ground truth growth rates
    true_growth = create_regional_growth(
        vertices, topo.faces, high_rate=0.8, low_rate=0.1, axis=2, threshold=0.0
    )

    params = SimParams(
        Kc=2.0, Kb=3.0, damping=0.9,
        skull_center=skull_center, skull_radius=skull_radius,
        skull_stiffness=100.0, carrying_cap_factor=4.0,
        tau=500.0, dt=0.02,
    )

    # Generate target folded surface
    n_sim_steps = 100
    target_verts, _ = create_target_folded(vertices, topo, true_growth, params, n_sim_steps)
    target_gi = float(gyrification_index(target_verts, topo, skull_radius))
    print(f"  Target GI: {target_gi:.3f}")

    # Initialize growth network
    key, subkey = jax.random.split(key)
    net = GrowthFieldNet(subkey, feature_dim=10, hidden=64)

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    weights = LossWeights(curv=1.0, gi=10.0, vertex=0.1)

    @eqx.filter_jit
    def train_step(net, opt_state):
        def loss_fn(net):
            features = extract_vertex_features(vertices, topo)
            vertex_growth = net(features)
            face_growth = growth_rates_to_faces(vertex_growth, topo)

            initial_state = make_initial_state(vertices, topo)
            final_state, _ = simulate(initial_state, topo, face_growth, params, n_sim_steps)

            return total_loss(
                final_state.vertices, target_verts, topo,
                skull_radius, target_gi, weights,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(net, eqx.is_array),
        )
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_opt_state, loss

    # Training loop
    n_epochs = 50
    print(f"\nTraining inverse model ({n_epochs} epochs)...")
    losses = []
    for epoch in range(n_epochs):
        net, opt_state, loss = train_step(net, opt_state)
        losses.append(float(loss))
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss = {float(loss):.6f}")

    # Evaluate recovered growth field
    features = extract_vertex_features(vertices, topo)
    recovered_vertex_growth = net(features)
    recovered_face_growth = growth_rates_to_faces(recovered_vertex_growth, topo)

    print(f"\n  True growth: min={float(true_growth.min()):.3f}, max={float(true_growth.max()):.3f}")
    print(f"  Recovered:   min={float(recovered_face_growth.min()):.3f}, max={float(recovered_face_growth.max()):.3f}")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inverse Training Loss")
    ax.set_yscale("log")
    plt.savefig("inverse_training_loss.png", dpi=150, bbox_inches="tight")
    print("  Saved inverse_training_loss.png")

    # Compare growth fields
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})
    plot_mesh(vertices, topo.faces, scalars=true_growth, title="True Growth", ax=ax1, cmap="YlOrRd")
    plot_mesh(vertices, topo.faces, scalars=recovered_face_growth, title="Recovered Growth", ax=ax2, cmap="YlOrRd")
    plt.savefig("growth_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved growth_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
