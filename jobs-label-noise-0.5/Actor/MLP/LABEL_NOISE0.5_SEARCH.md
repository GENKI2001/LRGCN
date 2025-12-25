# Experiment Configuration

**Dataset:** Actor

**Model:** MLP

**Total Jobs:** 6

## Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Hidden Channels | [32, 64, 128] |
| Num Layers | [1, 2] |

## Training Configuration

- **Epochs:** 1000
- **Early Stopping:** True
- **Patience:** 50
- **Number of Runs:** 20
