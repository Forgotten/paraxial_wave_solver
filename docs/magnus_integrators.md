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

The default `exponential_backend='adaptive_taylor'` chooses a Taylor degree
from 8, 12, 16, 20, and 24 and a number of repeated scaled actions. It minimizes
the planned operator-vector count subject to a per-action operator-tail budget
of `exponential_rtol / exponential_max_substeps`. The bound used is
`exp(theta)*theta**(m+1)/(m+1)!`, with `theta >= ||B/s||_2`. Defaults are
`exponential_rtol=1e-12` and `exponential_max_substeps=256`.

The spectral Laplacian bound is the largest transverse squared wavenumber;
periodic FD uses the absolute stencil row sum. No dense transverse matrix is
built. `exponential_plan` returns the degree, repeated actions, matvec count,
and capacity validity for diagnostics. A zero norm requires no operator actions.

This is a local truncation budget, **not a certified final relative error**:
errors can accumulate and be amplified, and float32 cannot meet float64-level
accuracy merely by tightening a Taylor budget. Integer selections are held
constant under AD. Fixed bounded loops and branches preserve reverse-mode
propagation through the selected polynomials.

Set `exponential_backend='fixed_taylor'` to retain the original 24-term method
at norm <= 1; this backend ignores `exponential_rtol`. Both backends return NaNs
when the available scaling capacity is insufficient. Reduce dz or raise
`exponential_max_substeps`. The selected implementation is conservative: it
uses an operator-norm tail bound, not the more sophisticated power-norm estimates
in [Al-Mohy and Higham's exponential-action algorithm](https://eprints.maths.manchester.ac.uk/1591/).

Run `python examples/magnus_actions.py` to compare work counts, accuracy, and
synchronized runtime. Runtime improvements depend on problem size and hardware;
unit tests enforce accuracy and reduced work, not a timing threshold.

Run `python examples/magnus_convergence.py` for a DOP853-reference convergence
study. Tests also cover spectral and FD discretizations with absorption/PML,
dense exponential references, invalid configurations, and directional gradients.
