# Experiment Configuration

**Dataset:** Cora

**Model:** LRGCN

**Total Jobs:** 90

## Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Hidden Channels | [32, 64, 128] |
| Num Layers | [1, 2] |
| Label Max Hops | [1, 2, 3] |
| Label Temperature | [0.125, 0.25, 0.5, 1.0, 2.0] |

## Training Configuration

- **Epochs:** 1000
- **Early Stopping:** True
- **Patience:** 50
- **Number of Runs:** 20
