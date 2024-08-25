# **BERT Quantization with PyTorch**

This repository contains a Jupyter Notebook that demonstrates the quantization of a BERT model using PyTorch. The notebook reduces the model size and enhances computational efficiency by converting weights to lower precision (4-bit quantization). This process optimizes the BERT model for deployment in resource-constrained environments.

## **Table of Contents**
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Notebook Overview](#notebook-overview)
- [Quantization Process](#quantization-process)
- [Results](#results)

## **Introduction**

Quantization is a technique that reduces the memory and computational requirements of deep learning models by using lower precision for weights and activations. This notebook focuses on quantizing a BERT model to 4-bit precision, significantly reducing its memory footprint while maintaining its performance.

### **Key Features:**
- Quantizes linear layers in a BERT model using a custom PyTorch module.
- Reduces model size and memory usage after quantization.
- Verifies the functionality of the quantized model with a forward pass example.

## **Getting Started**

To run the notebook, you need Jupyter Notebook (or Jupyter Lab) and the required Python libraries.

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/bert-quantization.git
cd bert-quantization
```
### **2. Install the Required Libraries**
Install the necessary dependencies using pip:

```bash
pip install torch transformers jupyter
```

### **3. Open the Jupyter Notebook**
Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```
## **Notebook Overview**

The notebook is structured as follows:

1. **Setup and Imports**: Import necessary libraries and set up the device (GPU or CPU) for model training and inference.
2. **Load Pre-trained BERT Model**: Load the BERT model (`bert-base-uncased`) and its tokenizer using the HuggingFace Transformers library.
3. **Define Custom Quantized Linear Layer**: Define a `QuantizedLinearLayer` class that quantizes the weights of linear layers to 4-bit precision.
4. **Replace Linear Layers with Quantized Layers**: Use the `replace_linearlayer` function to replace all linear layers in the BERT model with quantized versions.
5. **Calculate Model Size and Memory Footprint**: Calculate the model size before and after quantization to demonstrate the memory reduction achieved.
6. **Verify Quantized Model Functionality**: Perform a forward pass using the quantized model to ensure it functions correctly.
7. **Results and Observations**: Display and analyze the reduction in model size and memory footprint.

## **Quantization Process**

The quantization process involves the following steps:

1. **Initialize Quantized Weights**: Initialize weights in a lower precision format.
2. **Quantize Weights**: Convert original weights to a 4-bit format using a custom quantization method.
3. **Pack and Unpack Weights**: Efficiently pack weights to further reduce memory usage.
4. **Replace Standard Layers with Quantized Layers**: Replace the standard linear layers in the BERT model with quantized layers.
5. **Calculate and Compare Model Size**: Measure the model size before and after quantization.
6. **Run Forward Pass**: Verify the quantized model by running a forward pass with a sample input.

## **Results**

The quantization achieves significant memory reduction:

- **Original Model Size**: 417.64 MB
- **Quantized Model Size**: 91.39 MB

| **Model State**          | **Size (MB)** | **Memory Usage (GB)** |
|--------------------------|---------------|-----------------------|
| Original BERT Model      | 417.64        | 0.4379                |
| Quantized BERT Model     | 91.39         | 0.1389                |

Quantization reduces the model size by approximately 78%, demonstrating significant memory savings.
