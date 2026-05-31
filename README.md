# JAX_Finance
Demonstration of finance use cases for the JAX library and AutoDiff in particular

### 1.JAX_black_option_pricing.py

Implements Black Option pricing - gets "greeks" using jax.grad rather than analytically

### 2.SABR_JAX.py

Calibrate a vol surface using the SABR model  (Hagan et al 2002) using JAX to fit the model parameters to market vols </br>
`fx_data.py` contains EURUSD data for 21st May 2026 </br>
`fx_utils.py` functions to format broker-type quotes into strike/vol format to be used in the example for SABR_JAX.py</br>

