## Introduction

This project investigates neural-network design, hyperparameter tuning, and target representation for image-based deep-learning tasks using TensorFlow and Keras.

The project consists of two connected parts:

1. systematic comparison and tuning of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures on Fashion-MNIST and CIFAR-10;
2. development of a CNN-based model for reading the time from images of analog clocks.

### Image Classification and Hyperparameter Exploration

The first part compares MLP and CNN architectures on Fashion-MNIST and examines how model performance changes under different architectural and optimization choices.

The experiments vary:

- network depth and width;
- convolutional filter counts;
- activation functions;
- Adam, RMSprop, and SGD optimizers;
- dropout;
- L1, L2, and combined regularization;
- dense-layer size.

Validation data and early stopping are used during model selection to avoid using the test set for tuning.

The best Fashion-MNIST configurations are subsequently retrained from scratch on CIFAR-10 to investigate whether successful hyperparameter choices transfer to a more complex image dataset.

The best CNN achieves approximately **92.6% test accuracy on Fashion-MNIST** and **78.7% on CIFAR-10**, while the MLP models show a substantially larger performance decrease on CIFAR-10.

### Analog Clock Reading

The second part addresses a more specialized computer-vision problem: predicting the time shown in grayscale images of analog clocks.

The dataset contains 18,000 clock images with hour and minute labels. Images vary in rotation, viewing angle, lighting, and reflections, making the task more challenging than standard benchmark image classification.

A task-specific circular error metric is used:

**Circular error:** $E=\min(|T_{\mathrm{true}}-T_{\mathrm{pred}}|,\ 720-|T_{\mathrm{true}}-T_{\mathrm{pred}}|)$

where time is represented in minutes over a 12-hour cycle.

Several target formulations are compared:

- **24-class classification:** 30-minute intervals;
- **72-class classification:** 10-minute intervals;
- **720-class classification:** one class per minute;
- **regression:** a single continuous time value;
- **multi-head classification:** separate hour and minute classifiers;
- **hybrid multi-head modeling:** hour classification with minute regression.

All clock-reading models use a shared CNN feature extractor containing four convolutional blocks with Batch Normalization, Max Pooling, Dropout, and L2 regularization, followed by Global Average Pooling and task-specific output layers.

The multi-head models are implemented using the Keras Functional API. Different loss-weight ratios between the hour and minute prediction heads are evaluated to study how multi-task weighting affects final performance.

The best model uses separate classification heads for hour and minute prediction with an hour-to-minute loss-weight ratio of **2.5:1**, achieving a mean circular time error of approximately **0.86 minutes** on the held-out test set.

The project demonstrates the importance of both neural-network architecture and target representation: direct regression performs substantially worse because ordinary MSE does not account for the cyclical structure of time, while separating the task into hour and minute predictions provides a much more effective formulation.

---

## Summary

**Deep Learning – Image Classification & Analog Clock Reading:** Designed and systematically tuned MLP and CNN architectures (`TensorFlow`, `Keras`) using dropout, L1/L2 regularization, Batch Normalization, Early Stopping, and learning-rate scheduling; evaluated model transfer from Fashion-MNIST to CIFAR-10; and developed a multi-head CNN with weighted hour/minute classification losses, achieving a mean circular time error of **0.86 minutes** on analog-clock images.
