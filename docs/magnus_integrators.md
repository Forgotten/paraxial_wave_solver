# Fourth-order Magnus propagation

Select `SolverConfig(method='spectral', stepper='magnus4')` or use
`method='finite_difference'` with the existing periodic stencils. This linear
paraxial integrator evaluates the medium at the two Gauss nodes
`z + h*(1/2 ± sqrt(3)/6)`. It applies

```
exp(h*(a*A1 + b*A2)) exp(h*(b*A1 + a*A2)) psi
 a = (3 - 2*sqrt(3))/12, b = (3 + 2*sqrt(3))/12
```

where `Aj` is the full linear paraxial generator at a node. The rightmost
exponential acts first. This is the two-exponential commutator-free fourth-order
Magnus method; see the [primary implementation reference](https://doi.org/10.1088/1367-2630/14/10/105008).
Unlike freezing at one midpoint, these ordered actions retain fourth-order
accuracy for smooth, range-dependent noncommuting generators.

Kerr, square-root propagation, dealias filtering, and complex coordinate
stretching are rejected. A complex index and a local absorbing potential are
included in the generator, without an order-reducing absorber split. The
negative combination coefficient means strongly varying absorption does not
have an unconditional contractivity guarantee at intermediate stages.

Each exponential action uses 24 Taylor terms and repeated scaled actions with
an induced 2-norm bound of at most one. The spectral Laplacian bound is the
largest transverse squared wavenumber; periodic FD uses the absolute stencil
row sum. No dense transverse matrix is built. The fixed maximum loop length
preserves JAX reverse-mode differentiation, including physical parameter and
medium overrides. Integer scaling decisions are treated as constant by AD.

`exponential_max_substeps=256` bounds the number of scaled actions **per
exponential**. Exceeding it returns NaNs rather than an unchecked approximation;
reduce `dz` or increase the limit. This conservative algorithm can be expensive
on fine grids. It does not promise exact norm conservation at floating-point
precision. Its Taylor truncation is below roundoff at float64 for the scaled
bound; roundoff and nonnormal amplification can still limit accuracy.

Run `python examples/magnus_convergence.py` for a DOP853-reference convergence
study. Tests also cover spectral and FD discretizations with absorption/PML,
dense exponential references, invalid configurations, and directional gradients.
