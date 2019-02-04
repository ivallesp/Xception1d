# Changelog of the experiments run
This document contains the most important differences implemented in every experiment.

## Version 0.5
- Batchnorm to InstanceNorm (for conv layers) and LayerNorm (for dense layers)

## Version 0.4.1
- Back to ReLU activations
- Add weight decay (`lambda = 0.001`)
- Add learning rate scheduler (on plateau with `patience = 2` and `factor = 0.3`)

## Version 0.4
- Add swish activations
- Add dropout (`p = 0.75`)
