# Chapter 27 Solutions — Key Ideas

- a deterministic autoencoder does not automatically define a useful prior for random sampling.
- a VAE encoder predicts distribution parameters, commonly mu and log variance.
- reparameterization moves randomness into epsilon so gradients can flow through distribution parameters.
- the ELBO balances reconstruction likelihood and KL regularization.
- KL near zero can indicate posterior collapse rather than success.
