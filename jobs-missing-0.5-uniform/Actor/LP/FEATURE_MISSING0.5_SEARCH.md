# Experiment Configuration

**Dataset:** Actor

**Model:** LP

**Total Jobs:** 42

## Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Hidden Channels | [32, 64, 128] |
| Num Layers | [1, 2] |
| LP Alpha (LP) | [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999] |

## Training Configuration

- **Epochs:** 1000
- **Early Stopping:** True
- **Patience:** 50
- **Number of Runs:** 20
