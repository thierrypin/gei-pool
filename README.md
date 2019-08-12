
# Gait pooling experiments

Source code for the paper described in: [][].

Experiments were perfomed on the [OULP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLP.html) dataset. Since it uses different actors for training and testing, in this code the model evaluation is performed in two steps.

### Training

The training procedure is implemented in the `train.py` file. To run it for multiple models, folds and angles, use `train_grid.sh`.

### Testing

Similarly, the `test.py` file implements the testing procedure, while `test_grid.py` runs it for every combination of angles and folds.


