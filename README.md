[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# TSD: Transformer for Seizure Detection (Python Training & C Implementation)

This repository provides a complete framework for seizure detection using a Transformer-based model, as detailed in the paper **TSD: Transformers for Seizure Detection**. It includes:

* `TSD_PY`: A PyTorch-based environment for preprocessing the TUSZ dataset, training the Transformer model, and evaluating various approximation techniques (e.g., Softmax Taylor approximation, GeLU piecewise linear approximation) to assess their impact on model performance.
* `TSD_C`: A fixed-point ($FP16$) C implementation of the Transformer model, optimized for inference on low-power embedded systems, specifically the X-HEEP-based HEEPtimize platform. It includes both exact and approximated kernel versions of transformer.

---

## Repository Structure

```text
/
├── TSD_PY/         # Python code for training and evaluation
│   ├── data_processing/        
│   └── code/
└── TSD_C/          # C code for embedded inference
    └── sw/applications/
        ├── transformer/        # Exact C implementation
        ├── transformer_C_approx/ # Approximated C implementation
        └── transformer_kernels/  # Kernels for characterization
...
.gitignore
README.md
```
---
## Python Framework (`TSD_PY`)

This section contains the tools to preprocess data, train a new model, and run inference in Python.

### Environment Setup

Using `conda`, with channels `pytorch`, `default`, and `conda-forge`:

```bash
conda create --name tsd --file TSD_PY/requirement.txt
conda activate tsd
pip install -r TSD_PY/pyrequirements.txt
```
#### 1.  **Preprocessing**

This script filters signals, converts them to a bipolar montage, and extracts Short-Time Fourier Transform (STFT) features from the TUSZ v2.0.0 dataset.

1.  Navigate to the `TSD_PY/code` directory.
2.  Run the command:
    ```bash
    python tuh_dataset.py --data_directory /path/to/TUSZV2 --save_directory /path/to/save/preprocessed --data_type [train|dev|eval] --fft_amplitude [logarithm|absolute]
    ```

#### 2.  **Training**

Train the Vision Transformer (ViT) model on the preprocessed data.

1.  Navigate to the `TSD_PY/code` directory.
2.  Run the command:

    ```bash
    python best_model.py --save_directory /path/to/save/preprocessed --output_name <your_model_name> --train_approx <approx_type>
    ```

* The `<approx_type>` can be `smtaylor`, `gelupw`, `consmax`, `smtaylor_gelupw`, or `consmax_gelupw`. Omit this argument to train the baseline model without approximations.
* The trained model will be saved in a directory named `<your_model_name>` inside the `save_directory`.

#### 3.  **Inference**

Evaluate the trained model using either sample-based or event-based metrics.

1.  Navigate to the `TSD_PY/code` directory.
2.  Run the command:
    ```bash
    python inference.py --data_directory /path/to/save/preprocessed --model_folder <your_model_name> --timing [True|False]
    ```

* Set `--timing True` to benchmark the inference time over multiple runs.

---

## C Implementation (`TSD_C`)

This section contains the C implementation for running the trained model on an embedded target.

### Key Features

* **Fixed-Point:** Implemented in 16-bit fixed-point arithmetic for efficiency.
* **Embedded Target:** Designed to run on the [HEEPtimize](https://github.com/Ho3eintaji/HEEPtimize) platform (based on X-HEEP).
* **Optimized:** Weights are stored in flash and read into RAM layer-by-layer to minimize memory footprint.
* **Approximate Kernels:** The `transformer_C_approx` application includes modified kernels to accelerate inference time, corresponding to the approximations explored in the Python framework.

---

## Citation

If you use this repository in your research, please cite the following papers:
1. **Approximated Implementation**
    * Taji H, Miranda J, Peón-Quirós M, Atienza D. MEDEA: A Design-Time Multi-Objective Manager for Energy-Efficient DNN Inference on Heterogeneous Ultra-Low Power Platforms. arXiv preprint arXiv:2506.19067. 2025 Jun 23.

2.  **Original TSD Papers:**
    * Ma, Y., Liu, C., Ma, M.S., Yang, Y., Truong, N.D., Kothur, K., Nikpour, A. and Kavehei, O., 2023. TSD: Transformers for Seizure Detection. *bioRxiv*, pp.2023-01.

    * Amirshahi, Alireza & Toosi, Maedeh & Mohammadi, Siamak & Albini, Stefano & Schiavone, Pasquale & Ansaloni, Giovanni & Aminifar, Amir & Atienza, David. (2024). MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots. *arXiv:2408.01988*.
---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).