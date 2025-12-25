# Experiment Configuration

**Dataset:** Cora

**Model:** FISF

**Total Jobs:** 1296

## Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Hidden Channels | [32, 64, 128] |
| Num Layers | [1, 2] |
| FISF Num Iterations | [30, 50, 70, 90] |
| FISF Alpha | [0.1, 0.5, 0.9] |
| FISF Beta | [0.1, 0.5, 0.9] |
| FISF Gamma | [0.1, 0.5, 0.9] |
| FISF Mask Type | ['uniform', 'structural'] |

## Training Configuration

- **Epochs:** 1000
- **Early Stopping:** True
- **Patience:** 50
- **Number of Runs:** 20
