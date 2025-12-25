# Experiment Configuration

**Dataset:** Citeseer

**Model:** Goodie

**Total Jobs:** 72

## Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Hidden Channels | [32, 64, 128] |
| Num Layers | [1, 2] |
| LP Alpha (Goodie) | [0.9, 0.99, 0.999] |
| Goodie Lambda | [0.001, 0.01, 0.1, 1] |

## Training Configuration

- **Epochs:** 1000
- **Early Stopping:** True
- **Patience:** 50
- **Number of Runs:** 20
