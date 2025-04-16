# Surgical Tool Segmentation with Rarity-Aware Sampling and Compound Loss

This repository contains the implementation for semantic segmentation of surgical instruments using the SAR-RARP50 dataset. The project compares two loss configurations — Dice + CrossEntropy and Focal + CrossEntropy — with a focus on rare-class performance under limited computational resources.

---

## Key Features

- **Rarity-aware sampling** using `WeightedRandomSampler`
- Comparison of **Dice+CE** vs **Focal+CE** losses
- Evaluation on challenging, real-world surgical tool segmentation
- Fully reproducible with pretrained checkpoints
- Training analysis, metric plots, and qualitative visualizations

---

##  Project Structure

```bash
.
├── checkpoints/                # Best model (Dice+CE)
├── checkpoints_focal/         # Best model (Focal+CE)
├── scripts/                   # Evaluation + plotting scripts
│   ├── evaluate_test.py
│   └── plot_training_metrics.py
├── src/                       # Core logic
│   ├── dataset.py
│   ├── loss.py
│   ├── metric.py
│   ├── model.py
│   ├── sampler.py
│   ├── train.py
│   └── utils.py
├── Data_Extraction.ipynb      # Dataset analysis + class imbalance
├── README.md
└── .gitignore
```
### Install Dependencies

```bash
conda create -n surgery-seg python=3.9
conda activate surgery-seg
pip install -r requirements.txt

```
## Training

All training is done using the script `src/train.py`.

By default, the training script uses the **Dice + CrossEntropy** loss.

### Run Training (Default: Dice + CE)

```bash
python src/train.py
```
## To switch to Focal + CrossEntropy
```bash
# Replace this
loss_fn = DiceCrossEntropyLoss()

# With this
from loss import FocalCrossEntropyLoss
loss_fn = FocalCrossEntropyLoss()
```

---

## Future Development

While the current version of this project is fully functional, there are several improvements planned for better usability and extensibility:

- **YAML-based configuration**: Integrate a config system (e.g., with `PyYAML` or `omegaconf`) to allow users to change dataset paths, model settings, loss functions, and training hyperparameters without modifying code.

- **Command-line interface (CLI)**: Replace hardcoded parameters with `argparse`-based CLI arguments, making it easy to train or evaluate with different settings.

- **Single-image inference**: Add a script to run inference on a single custom image outside the test set (for demo/testing).

- **WandB or TensorBoard integration**: Add support for real-time metric tracking during training.

If time and resources allow, these features will be included in a future version of this repository.

