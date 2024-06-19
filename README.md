<p align="center">
  <h1 align="center">FingerFlexion</h1>
  This project was to develop an algorithm to decode finger movements from ECoG signals. The final model achieved a correlation score of 0.893 on the hidden test set.
</p>

## Abstract 
Motor brain-computer interface (BCI) development relies critically on neural time series decoding algorithms. Recent advances in deep learning architectures allow for automatic feature selection to approximate higher-order dependencies in data. This project presents a convolutional encoder-decoder architecture adapted for finger movement regression on electrocorticographic (ECoG) brain data. The model achieved a correlation coefficient between true and predicted trajectories, demonstrating the potential for developing high-precision cortical motor brain-computer interfaces.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Summary](#algorithm-summary)
- [Detailed Explanation](#detailed-explanation)
- [Model Architecture](#model-architecture)
- [Results and Future Work](#results-and-future-work)
- [Usage](#usage)
- [References](#references)

## Introduction
This project was developed as part of the BE5210 course, where the goal was to develop an algorithm to decode finger movements from ECoG signals. The final model achieved a correlation score of 0.567 on the testing dataset.

## Algorithm Summary
The signal processing pipeline and training algorithm were inspired by several published articles. The key steps include:
- Normalization of ECoG signals
- Filtering using Hanning FIR and notch filters
- Continuous wavelet transform for feature extraction
- Downsampling for computational efficiency
- Training a 1D U-Net to map ECoG features to data glove signals

## Detailed Explanation

### Processing ECoG Signal
1. **Normalize Signal:** Subtract mean and divide by standard deviation.
2. **Filtering:** Apply Hanning FIR filter (20-300 Hz) and notch filter (60 Hz and harmonics).
3. **Wavelet Transform:** Perform continuous wavelet transform with Morlet wavelets.
4. **Downsample:** Reduce sampling rate from 1000 Hz to 100 Hz.
5. **Split Data:** 80% for training, 20% for validation.
6. **Time Shift and Scaling:** Account for signal delays and scale using RobustScaler.

### Processing Glove Signal
1. **Interpolate Data:** Downsample to 25 Hz, then upsample to 100 Hz using cubic splines.
2. **Split Data:** Similar to ECoG data.
3. **Time Shift and Scaling:** Adjust for delays and scale using MinMaxScaler.

## Model Architecture
- **1D U-Net:** Consists of an encoder and decoder with skip connections.
- **Training:** Adam optimizer with learning rate and weight decay. Includes validation callbacks and checkpointing.
- **Validation:** Gaussian smoothing and correlation calculation.

### Model Details
- **Encoder:** Reduces spatial dimensions of input data.
- **Decoder:** Upsamples data back to original dimensions.
- **Skip Connections:** Preserve high-frequency details during encoding and decoding.

## Results and Future Work
- **Performance:** Achieved a correlation score of 0.893 on the hidden testset.
- **Improvements:** Future work could explore 2D U-Net architectures, refine preprocessing techniques, and integrate transfer learning.

## Usage
1. **Preprocess Data:**
   ```python
   from preprocess import preprocess_ecog, preprocess_dg
   ecog_train, ecog_val = preprocess_ecog(ecog_data, low_freq, high_freq, sample_rate, shift_time, downsample_f_s, num_wavelets)
   dg_train, dg_val = preprocess_dg(dg_data, super_sf, true_sf, desired_sf, shift_time)
   ```
2. **Define Dataset and DataLoader:**
   ```python
   from dataset import Dataset, Datamodule
   data_module = Datamodule(patient=1, batch_size=128, window_n=256, stride_n=1)
   ```
3. **Train Model:**
   ```python
   from model import EDModel_Skip, Runner
   model = EDModel_Skip(n_electrodes=30, n_freqs=16, n_channels_out=21)
   runner = Runner(model)
   runner.fit(data_module)
   ```
4. **Evalute Model:**
   ```python
   runner.evaluate(validation_data)
   ```
