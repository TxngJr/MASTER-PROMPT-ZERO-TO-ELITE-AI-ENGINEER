# Scientific ML Solutions

1. A surrogate is a cheaper approximation to an expensive simulator/measurement process.
2. A physics residual is the mismatch after substituting a learned function into the governing equation.
3. A collocation point is a coordinate where the equation residual is evaluated during training/evaluation.
4. Nondimensionalization rescales variables using characteristic units to remove/normalize physical dimensions and improve conditioning.
5. Data loss matches observations; physics loss penalizes governing-equation violations. Their weights define an optimization trade-off.
6. A solution can satisfy an equation family yet violate the specific problem without its boundary/initial conditions.
7. For du/dt=-k u, residual r=du_theta/dt + k u_theta.
8. Interpolation predicts within supported parameter regimes; extrapolation goes outside them and requires separate evidence.
9–11. See `src/scientific_ml.py`; verify finite-difference order assumptions and units.
12. Generate simulator pairs (parameter,output), fit a polynomial basis on training regimes and evaluate held-out regimes.
13. Compare derivative against an analytic derivative across grid spacing and show truncation error decreases with refinement until numerical limits matter.
14. Group split by trajectory/parameter so correlated samples from one physical run cannot appear in both train and test.
15. Express variables in characteristic scales and compare gradient/loss magnitudes under otherwise matched settings.
16. They answer different questions: equation satisfaction at sampled points versus agreement with the reference solution.
17. Include PDE/ODE residual, both boundary conditions and optional data terms with documented weights.
18. Hold test regimes fixed and compare solution error/residual under multiple seeds/data sizes.
19. A neural operator learns a mapping between input functions/fields and output functions/fields rather than only a fixed-dimensional pointwise mapping.
20. Record governing equations, units, domains, boundary/initial conditions, data generation, reference solver, splits, architecture, residual/solution metrics, compute and limitations.
