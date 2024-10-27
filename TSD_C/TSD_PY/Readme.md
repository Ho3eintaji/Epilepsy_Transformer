# Transformer for Seizure Detection

This repository contains code to train a Transformer model for seizure detection based on the paper:

**TSD: Transformers for Seizure Detection**  
Yongpei Ma, Chunyu Liu, Maria Sabrina Ma, Yikai Yang, Nhan Duy Truong, Kavitha Kothur, Armin Nikpour, Omid Kavehei  
bioRxiv 2023.01.24.525308; doi: https://doi.org/10.1101/2023.01.24.525308

## Dataset

The model is trained using the TUSZ V2.0.0 dataset. You need to download this dataset before running the code.

## Preprocessing

To preprocess the TUSZ v2.0.0 dataset:

1. Navigate to the `TSD/code` directory.
2. Run the following command:

    ```bash
    python tuh_dataset.py --data_directory /path/to/the/TUSZV2/dataset --save_directory /path/to/save/the/preprocessed/signals
    ```

This script applies filtering to the input signals, converts them to bipolar montage, and extracts STFT files. Note that this process may take some time.

## Training

To train the Transformer model:

1. Navigate to the `TSD/code` directory.
2. Run the following command:

    ```bash
    python best_model.py --data_directory /path/to/the/TUSZV2/dataset --save_directory /path/to/save/the/preprocessed/signals
    ```

The trained model will be saved in a directory called `out_model` inside the `save_directory`.

## Notes

- If you're not using a GPU, change the `device` variable in the code from 'cuda' to 'cpu'.
- The `TSD/code/TUSZv2_info.json` file contains pre-extracted labeling information used in preprocessing.

## Citation

If you use this repository, please cite the following papers:

1. Amirshahi, Alireza & Toosi, Maedeh & Mohammadi, Siamak & Albini, Stefano & Schiavone, Pasquale & Ansaloni, Giovanni & Aminifar, Amir & Atienza, David. (2024). MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots. 10.48550/arXiv.2408.01988.

2. Dan J, Pale U, Amirshahi A, Cappelletti W, Ingolfsson TM, Wang X, et al. SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of electroencephalography-based automated seizure detection algorithms. Epilepsia. 2024; 00: 1–11. https://doi.org/10.1111/epi.18113

3. Amirshahi, A., Dan, J., Miranda, J.A., Aminifar, A., and Atienza, D.. (2024). FETCH: A Fast and Efficient Technique for Channel Selection in EEG Wearable Systems. Proceedings of the fifth Conference on Health, Inference, and Learning, 248:397-409.