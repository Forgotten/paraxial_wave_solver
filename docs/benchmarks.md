# Accuracy versus cost

Run `python examples/solver_benchmarks.py --output /tmp/solver_benchmarks.json`.
The small default grid makes independent references affordable; increase the
transverse size through `run_case(n=...)` for scaling studies.

The suite separates four physical problems: smooth range-dependent paraxial
media, Kerr nonlinearity, absorption, and full-index near-cutoff propagation.
Paraxial references use tight-tolerance SciPy DOP853 on the same spectral grid;
the constant full-index reference uses a dense Hermitian eigendecomposition.
Different physical models are never ranked as substitutes for each other.

Each method refines dz at fixed grid and total range. The report selects the
fastest measured configuration **per method and problem** meeting the requested
field-error target. Unreached targets and nonfinite results remain explicit.
This is temporal accuracy against the discretized equation, not continuum
spatial accuracy. The reference tolerance must remain smaller than errors of
interest; results at the reference/roundoff floor should not establish order.

Compilation is measured separately. Execution is warmed, synchronized, and
reported as a median. Compiler-reported temporary, argument, and output buffer
sizes are recorded separately; they are not process RSS or measured peak device
memory. Unsupported memory reports are null. FFT counts are static counts for
RK4, ETDRK4, and Strang; Magnus counts use the selected action plans evaluated
outside the timed region. Other iterative action counts are null,
not guesses. Timings exclude diagnostic instrumentation and reference solves.
Hardware, JAX version, and devices are saved with the report. Do not compare
unsynchronized timings or enforce speed thresholds in unit tests.

Tests verify reference conservation/decay, numerical convergence, metadata,
and accuracy selection. Runtime is deliberately not a pass/fail criterion.
