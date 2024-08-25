# **BERT Quantization with PyTorch**

This repository contains a Jupyter Notebook that demonstrates the quantization of a BERT model using PyTorch. The notebook reduces the model size and enhances computational efficiency by converting weights to lower precision (4-bit quantization). This process optimizes the BERT model for deployment in resource-constrained environments.

## **Table of Contents**
- [Introduction](#introduction)
- [Background](#background)
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

## **Background**
### **BERT Layers and Theory**

**BERT** (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language representation model developed by Google. It is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right contexts in all layers. This makes BERT particularly effective for a wide range of natural language understanding tasks.

### **Architecture of BERT:**

BERT is based on the Transformer architecture and consists of the following key components:

1. **Input Embedding Layer:**
   - Converts input tokens (words or sub-words) into dense vector representations.
   - The input embeddings are the sum of:
     - **Token embeddings:** Represent each token in the vocabulary.
     - **Position embeddings:** Capture the position of each token in the sequence.
     - **Segment embeddings:** Distinguish different sentences or segments in tasks like sentence pair classification.

2. **Multiple Encoder Layers:**
   - BERT uses a stack of Transformer encoder layers. Each encoder layer consists of:
   
   - **Self-Attention Mechanism:** 
     - Allows each token to attend to every other token in the input sequence, capturing contextual relationships.
     - Computes a weighted sum of the values, with the weights determined by a similarity score (typically using scaled dot-product attention).

   - **Feed-Forward Neural Network:**
     - Applies a fully connected feed-forward network to the output of the self-attention layer.
     - Typically consists of two linear transformations with a ReLU activation in between.

   - **Layer Normalization and Residual Connections:**
     - Ensures stability and improves training by normalizing the output of each sub-layer and adding residual connections.

3. **Output Layer:**
   - The output of BERT is a sequence of hidden states corresponding to each input token.
   - These hidden states can be used for various NLP tasks such as text classification, named entity recognition, and question answering.

![image](https://github.com/user-attachments/assets/6c288bbf-d499-49ba-a5e7-fa666c8c9509)

*Figure 1: BERT Architecture showing input embeddings, multi-layer encoders, and output representations*

---

### **Theory Behind Quantization:**

Quantization is the process of mapping input values from a large set (like floating-point numbers) to output values in a smaller set (like integers). In deep learning, quantization reduces the precision of the weights and activations of a model to reduce its size and computational requirements, making it suitable for deployment in environments with limited resources.

#### **Types of Quantization:**

1. **Post-Training Quantization:** 
   - Converts a pre-trained model's weights to lower precision without additional training.

2. **Quantization-Aware Training:**
   - Simulates quantization during training to improve the accuracy of the quantized model.

#### **Benefits of Quantization:**

- **Reduced Model Size:** 
  - Lowers the memory footprint, making the model easier to deploy on edge devices.

- **Faster Inference:** 
  - Reduces the computational complexity, resulting in faster inference times.

- **Energy Efficiency:** 
  - Decreases power consumption, beneficial for mobile and embedded applications.

### **Quantization Process**

The quantization process involves the following steps:

1. **Initialize Quantized Weights:** 
   - Initialize weights in a lower precision format.

2. **Quantize Weights:** 
   - Convert original weights to a 4-bit format using a custom quantization method.

3. **Pack and Unpack Weights:** 
   - Efficiently pack weights to further reduce memory usage.

4. **Replace Standard Layers with Quantized Layers:** 
   - Replace the standard linear layers in the BERT model with quantized layers.

5. **Calculate and Compare Model Size:** 
   - Measure the model size before and after quantization.

6. **Run Forward Pass:** 
   - Verify the quantized model by running a forward pass with a sample input.



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
