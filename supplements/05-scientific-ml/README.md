# Supplement 05 — Scientific Machine Learning

Scientific ML combines physical/scientific structure with data-driven function approximation.

## Learning objectives

- distinguish data-only regression from equation-constrained learning;
- understand residual minimization for differential equations;
- understand Physics-Informed Neural Network (PINN) concepts;
- understand boundary/initial-condition losses;
- understand nondimensionalization and scaling;
- understand neural-operator concepts;
- build surrogate models and quantify error;
- design train/validation tests that respect parameter/time/space structure.

## Residual-based learning

Suppose a differential equation is:

~~~text
du/dt = f(u,t)
~~~

A differentiable model (u_	heta(t)) can be trained with residual:

~~~text
r(t) = d u_theta(t)/dt - f(u_theta(t), t)
L_physics = mean(r(t)^2)
~~~

Add initial/boundary/data terms:

~~~text
L =
lambda_data * L_data
+ lambda_physics * L_physics
+ lambda_bc * L_boundary
~~~

Loss weights affect optimization and must be reported.

## PINN mental model

~~~text
coordinates/parameters
→ neural function approximation
→ automatic derivatives
→ PDE/ODE residual
→ physics + boundary + data losses
~~~

A low residual at collocation points does not guarantee global solution accuracy. Validate against analytic/numerical references where available and inspect residuals away from training points.

## Nondimensionalization

Poor scale differences between variables/terms can make optimization difficult. Rescale variables using characteristic units and report the transformation so predictions can be mapped back to physical units.

## Neural operators

A standard neural network often maps a finite vector to another vector/value. Neural-operator families aim to learn mappings between functions/fields, such as boundary/initial conditions to solution fields. Examples include Fourier Neural Operator concepts and DeepONet-style branch/trunk ideas.

## Surrogate modeling

A surrogate approximates an expensive simulator. Evaluate:
- interpolation vs extrapolation;
- parameter-space coverage;
- conservation/constraint errors;
- uncertainty/error maps;
- speedup under the same requested output fidelity.

## Common mistakes

- random point splits that leak the same trajectory/parameter regime into train and test;
- reporting PDE residual but not solution error;
- mixing dimensional and nondimensional quantities;
- ignoring boundary conditions;
- overclaiming extrapolation from interpolation-only tests.
