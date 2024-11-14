# Training the Pix2Pix Model for SketchifyAI

This directory contains the necessary files and instructions to train a Pix2Pix model for converting sketches into realistic images. The training process uses paired images of sketches and realistic photos, which are stored in the `datasets/` folder.

## Directory Contents

- **train_pix2pix.py**: Main training script for the Pix2Pix model.
- **pix2pix_model.py**: Defines the architecture for the Pix2Pix generator and discriminator models.

## Prerequisites

1. **Python 3.6+**: Ensure Python 3 is installed.
2. **Dependencies**: Install the required libraries listed in the `backend/requirements.txt` file.

   ```bash
   pip install -r ../backend/requirements.txt