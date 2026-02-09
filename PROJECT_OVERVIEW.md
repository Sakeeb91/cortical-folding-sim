# Differentiable Cortical Folding Simulator

## What This Project Is

This is a JAX-based differentiable physics simulator that models how the human cerebral cortex folds during fetal development. It takes a smooth sphere (representing the early fetal brain surface) and simulates the mechanical process of cortical buckling — producing the characteristic ridges (gyri) and grooves (sulci) that define the brain's architecture.

What makes this simulator different from traditional finite-element brain models is that it is **end-to-end differentiable**. Every operation — from force computation to time integration to growth — flows through JAX's automatic differentiation. This means you can not only run the simulation forward (growth parameters → folded surface), but also solve the **inverse problem**: given an observed brain shape, recover the growth parameters that produced it via gradient descent.

The simulator implements the mechanical model described in Nie et al. (2010), "A Computational Model of Cerebral Cortex Folding" (*Journal of Theoretical Biology*), extended with a neural network growth field predictor and modern differentiable programming techniques.

---

## The Real-World Problem

### Why Brains Fold

The human cortex packs approximately 2,600 cm² of neural surface area into a skull that could only fit about 1,800 cm² if the cortex were smooth. This folding isn't random — it emerges from a mechanical instability. During weeks 12–24 of gestation, the outer cortical plate (gray matter) expands tangentially faster than the underlying white matter it's attached to. This growth mismatch causes the cortical sheet to buckle, the same physics that wrinkles the skin of a drying fruit.

The specific folding pattern is determined by:

- **Cortical thickness** — thinner cortex buckles at higher spatial frequency (more, smaller folds)
- **Relative stiffness** — the ratio of cortical to subcortical stiffness sets the fold wavelength
- **Growth rate differential** — greater mismatch produces deeper sulci
- **Initial surface geometry** — small undulations seed where folds nucleate

### When Folding Goes Wrong

Abnormal cortical folding is a hallmark of several severe neurological conditions:

**Lissencephaly** (smooth brain) — 12–40 per million births. The brain develops few or no folds due to defective neuronal migration. Over 90% of affected individuals develop epilepsy, typically in the first year of life. Mechanically, the cortex is too thick and stiff to buckle. Mutations in LIS1 and DCX genes are the most common cause.

**Polymicrogyria** (excessive small folds) — 2.3 per 10,000 children. The cortex develops too many abnormally small folds. Epilepsy occurs in 54–87% of patients, with 56% being drug-resistant. Mechanically, the cortex is abnormally thin, causing it to buckle at a higher spatial frequency. Congenital CMV infection is the most common cause.

**Schizophrenia** — shows age-dependent gyrification abnormalities. Young patients show increased gyrification in frontal regions; older patients show decreased gyrification. Since cortical folding is established prenatally, these patterns serve as trait biomarkers for the developmental origins of the disease.

**Autism spectrum disorder** — children with ASD show increased local cortical gyrification that decreases rapidly during adolescence. Altered frontal gyrification correlates with disrupted functional connectivity.

**Epilepsy** — malformations of cortical development are a leading cause of drug-resistant epilepsy more broadly.

The core clinical challenge is that we can observe these abnormal folding patterns on MRI, but we cannot directly measure the growth parameters (thickness, stiffness, growth rate) that caused them. The brain developed months or years before the scan. This is an inverse problem — and it's exactly what a differentiable simulator is built to solve.

---

## How It Works

### The Physics

The simulator represents the cortical surface as a triangulated mesh and evolves it under four forces:

1. **Elastic force** — spring-like resistance along mesh edges. When an edge stretches beyond its rest length, a restoring force pulls it back. Computed via edge-scatter: per-edge forces accumulated to vertices using `jnp.zeros().at[idx].add()`.

2. **Bending force** — resistance to curvature change. Computed as the negative gradient of bending energy (Kb * sum of (H - H₀)² * area) via `jax.grad`, making it automatically differentiable without manual derivation.

3. **Growth** — rest areas of mesh faces grow logistically (dA/dt = A * m * (1 - A/k) * dt), and rest edge lengths scale proportionally to the square root of area growth. The lag between growing rest geometry and actual geometry creates elastic mismatch — the mechanical stress that drives buckling.

4. **Skull constraint** — a soft penalty that pushes vertices inward when they exceed the skull radius. The skull confines expansion and forces the growing cortex to fold rather than simply expand outward.

Time integration uses the explicit Newmark scheme with velocity damping. The full simulation runs via `jax.lax.scan` with `jax.checkpoint` for O(1) memory during backpropagation through hundreds of timesteps.

### The Inverse Problem

The inverse pipeline works as follows:

```
Observed brain MRI → target surface
                         ↓
Initial sphere → GrowthFieldNet(geometry features) → per-vertex growth rates
                         ↓
                    simulate(growth) → predicted surface
                         ↓
                 loss(predicted, target) → jax.grad → update network
```

A neural network (Equinox MLP) takes local geometry features at each vertex — position, normal, mean curvature, Gaussian curvature, area — and predicts a spatially-varying growth rate. The predicted growth field drives the forward simulation, and the loss between the simulated and observed surfaces backpropagates through the entire simulation to update the network weights.

This is only possible because every component is differentiable. A traditional FEM solver (ABAQUS, COMSOL) is a black box — you can run it forward but cannot backpropagate through it.

---

## What Differentiability Enables

| Capability | Traditional Simulation | This Simulator |
|---|---|---|
| Forward prediction | Yes | Yes |
| Inverse parameter estimation | Brute-force search over parameter space | Direct gradient-based optimization |
| Sensitivity analysis | One perturbation at a time | All sensitivities computed simultaneously |
| Neural network integration | Not possible (no gradient flow) | Seamless backpropagation through physics |
| Patient-specific fitting | Computationally prohibitive | Feasible via gradient descent |
| Scaling to many parameters | Cost grows linearly with parameter count | Cost independent of parameter count |

The gradient tells you exactly how to adjust each of hundreds or thousands of spatially-varying growth parameters to better match an observed brain shape. Without differentiability, you'd need finite-difference approximations (one forward simulation per parameter) or expensive sampling methods.

---

## Value Contribution

### For Neuroscience Research

**Mechanistic hypothesis testing.** The simulator lets researchers ask precise questions: "What minimal change in growth rate transforms normal folding into lissencephaly?" Instead of manually trying parameter combinations, gradient-based optimization finds the answer directly. Nie et al. showed that reducing growth rate from m=0.002 to m=0.001 in specific regions altered both local and global folding patterns — the differentiable version can systematically identify which parameters matter most.

**Bridging scales.** Molecular biology identifies genes (LIS1, DCX, Trnp1) that affect neuronal migration and proliferation. Clinical imaging reveals abnormal folding patterns. This simulator connects the two: gene expression → growth rate field → mechanical folding → observable morphology. By fitting the simulator to patient data, researchers can infer which upstream biological processes are disrupted.

### For Clinical Medicine

**Early diagnosis.** Fetal MRI can reveal cortical morphology in utero, but interpreting subtle folding abnormalities requires quantitative analysis. A fitted simulator could flag cases where the inferred growth parameters fall outside normal ranges, potentially catching conditions like polymicrogyria before birth when the folding pattern alone is ambiguous.

**Disease subtyping.** Two patients with the same diagnosis (e.g., polymicrogyria) may have different underlying growth parameter profiles. The inverse problem can distinguish between cases caused by globally thin cortex, regionally reduced growth, or altered stiffness ratios — information relevant for prognosis and treatment planning.

**Toward digital twins.** The broader medical field is moving toward patient-specific computational models (digital twins) for treatment planning. Cardiac digital twins are already in clinical use. Brain development models are earlier in this pipeline, but a differentiable simulator that can be fit to individual patient data is a necessary foundation.

### For Computational Science

**Differentiable physics methodology.** This project demonstrates the pattern of embedding a differentiable physical simulator inside a machine learning training loop — a technique with applications far beyond neuroscience. The same architecture (lax.scan + checkpoint + neural parameter field + gradient-based inverse) applies to any physical system where you want to infer spatially-varying material properties from observed outcomes: tissue mechanics, fluid dynamics, material science.

**Reproducible baseline.** Existing cortical folding simulations typically use commercial FEM packages (ABAQUS, COMSOL) that are expensive, closed-source, and difficult to modify. A self-contained JAX implementation with clear physics, open source code, and a test suite provides a reproducible baseline that others can build on.

---

## Technical Summary

```
Language:       Python 3.11+
Core framework: JAX (autodiff, lax.scan, vmap, checkpoint)
Neural network: Equinox (JAX-native modules)
Optimizer:      Optax (Adam for inverse training)
Mesh:           trimesh (icosphere generation)
Visualization:  matplotlib (3D trisurf)

Mesh representation:  MeshTopology NamedTuple (static connectivity)
Simulation state:     SimState NamedTuple (vertices, velocities, rest geometry)
Integration:          Explicit Newmark with velocity damping
Growth model:         Logistic area growth + rest-length scaling
Differentiability:    Full backprop through T-step simulation via lax.scan + checkpoint
```

---

## References

- Nie, J., Li, G., & Shen, D. (2010). A computational model of cerebral cortex folding. *Journal of Theoretical Biology*, 264(2), 467–478.
- Tallinen, T., et al. (2014). Gyrification from constrained cortical expansion. *PNAS*, 111(35), 12667–12672.
- Bayly, P. V., et al. (2013). A cortical folding model incorporating stress-dependent growth explains gyral wavelengths and stress patterns in the developing brain. *Physical Biology*, 10(1), 016005.
- Budday, S., et al. (2014). A mechanical model predicts morphological abnormalities in the developing human brain. *Scientific Reports*, 4, 5644.
- Holland, M. A., et al. (2018). Mechanics of cortical folding: stress, growth and stability. *Philosophical Transactions of the Royal Society B*, 370(1632).
