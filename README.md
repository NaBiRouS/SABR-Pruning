# SABR-Pruning: Sparsity-Accuracy Balanced Regularization
Adaptive Sparsity-Accuracy Balanced Regularization and pruning for neural networks

## Module Information

- **Module Name:** Réseaux de neurones et Apprentissage Automatique
- **Year:** Master Degree in Machine Learning and Computer Vision
- **University:** USTHB
- **Students:**
  - Boumedine Billal (181837068863)
  - Tolba Adel (212131030403)

> ### 🔍 **IMPORTANT: Refer to the Implementation Notebook**
>
> The [**SABR Implementation Notebook**](./SABR_Implementation.ipynb) contains:
>
> - Step-by-step algorithm explanation
> - Detailed training process visualization
> - Complete set of result plots and charts
> - In-depth analysis of model performance
> - Formulas and calculations for computational savings
>
> **Note:** The notebook should be your primary reference for understanding the implementation details.


## Project Goal

The main goal of this project is to implement and evaluate SABR (Sparsity-Accuracy Balanced Regularization), an innovative neural network pruning approach that balances model sparsity and accuracy through adaptive regularization. By inducing sparsity in deep neural networks while preserving their predictive performance, we can significantly reduce computational requirements and model size, making deployment on resource-constrained edge devices more practical.

Traditional deep learning models typically contain millions of parameters, requiring substantial computational resources and memory. This makes deployment on embedded systems, IoT devices, and mobile phones challenging. SABR addresses this challenge by strategically reducing model complexity while maintaining performance, enabling complex AI models to run efficiently on edge devices with limited processing power and memory.

## Implementation Details

### Core Technical Innovations

The SABR algorithm introduces four key innovations:

1. **Layer-specific Lambda Initialization**: Instead of using a uniform regularization strength across all layers, SABR initializes each layer's regularization parameter (lambda) based on the statistical properties of that layer's weights. Each lambda is calculated as `λ_layer = teta1 * std(weights)`, where teta1 is a hyperparameter and std is the standard deviation of the layer's weights.

2. **Adaptive Lambda Adjustment**: During training, lambda values are dynamically adjusted based on validation performance:
   - If validation accuracy drops below a threshold, lambda values are decreased to reduce regularization pressure
   - If validation accuracy remains stable, lambda values are increased to enhance sparsity
   - This adaptive approach maintains a careful balance between sparsity and accuracy

3. **Progressive Parameter Freezing**: After a layer has been stable for a specified number of epochs (k_epoch), its most sparse parameter is frozen, preventing further changes. This gradual freezing process:
   - Focuses learning on parameters that still have significant information content
   - Progressively stabilizes the model toward a sparse representation
   - Reduces computation during backpropagation as training progresses

4. **Post-training Pruning**: After training, a final pruning step converts near-zero weights to exact zeros based on layer-specific thresholds, enabling hardware and software optimizations for sparse neural networks.

### Three-Phase Methodology

Our implementation follows a three-phase approach:

#### Phase 1: Base Model Training

- Train a neural network using standard techniques without regularization
- Establish a baseline for model performance
- Obtain a well-trained model that captures the essential patterns in the data

#### Phase 2: SABR Regularization

- Apply adaptive L1 regularization with layer-specific lambda values
- Dynamically adjust regularization strength based on validation accuracy
- Progressively freeze parameters, starting with the most sparse ones
- Induce sparsity while maintaining model performance

#### Phase 3: Post-Training Pruning

- Apply thresholding to convert near-zero weights to exact zeros
- Use standard deviation-based thresholds for optimal results
- Evaluate the pruned model to confirm it maintains acceptable accuracy
- Calculate sparsity and computational savings

## Project Structure

The SABR-Pruning project follows a modular structure:

```
SABR-Pruning/
│
├── sabr/                        # Core SABR algorithm implementation
│   ├── sabr.py                  # Main SABR class for adaptive regularization
│   ├── lambda_calculator.py     # Lambda calculation for layer-specific regularization
│   ├── pruning.py               # Pruning functionality and sparsity metrics
│   └── __init__.py              # Package initialization
│
├── utils/                       # Utility functions
│   ├── data_handler.py          # Dataset loaders and preprocessing
│   ├── training_utils.py        # Training and evaluation helpers
│   ├── visualization.py         # Plotting tools for results analysis
│   └── __init__.py              # Package initialization
│
├── notebook_utils/              # Notebook-specific utilities
│   ├── model_utils.py           # Model building and comparison tools
│   ├── operations_utils.py      # Computational cost calculation
│   ├── testing_utils.py         # Visual testing utilities
│   └── visualization_utils.py   # Enhanced visualization tools
│
├── SABR_Implementation.py    # Complete implementation with detailed explanations
├── main.py                      # Complete training pipeline script
├── apply_pruning_simplified.py  # Standalone pruning script
├── realtime_recognition.py      # Real-time hand gesture recognition application
└── README.md                    # Project documentation
```

## Results and Benefits

Our implementation of SABR achieves impressive results in balancing model sparsity and accuracy. We evaluated the approach on two popular architectures: VGG11 and MobileNetV2.

### VGG11 Results

| Phase                 | Accuracy | Sparsity | Parameter Reduction |
| --------------------- | -------- | -------- | ------------------- |
| Base Model            | 99.96%   | 0.00%    | 0%                  |
| SABR Regularization   | 100.00%  | 85.76%   | 85.76%              |
| Post-Training Pruning | 99.14%   | 97.66%   | 97.66%              |

The final pruned VGG11 model achieved:

- **97.66%** sparsity (only 2.34% of weights remain non-zero)
- **93.72%** reduction in computational operations
- Only **515,534** non-zero parameters out of **22,069,952** total parameters
- Negligible accuracy drop of **0.29%** from the base model

![VGG11 Accuracy vs Sparsity](/notebook_results/accuracy_vs_sparsity.png)

The above chart shows how our approach achieves significantly higher sparsity with minimal accuracy loss compared to standard methods.

![VGG11 Computational Savings](/notebook_results/pruned_model/current_modelcomputational_savings_overall.png)

This chart demonstrates the dramatic reduction in computational operations achieved by our pruned model.

![VGG11 Layer-wise Savings](/notebook_results/pruned_model/current_modelcomputational_savings_by_layer.png)

The layer-wise analysis shows that our method effectively prunes across all layers of the network, with particularly high sparsity in the computationally intensive layers.

### MobileNetV2 Results

| Phase                 | Accuracy | Sparsity | Parameter Reduction |
| --------------------- | -------- | -------- | ------------------- |
| Base Model            | 100.00%  | 0.00%    | 0%                  |
| SABR Regularization   | 99.96%   | 86.25%   | 86.25%              |
| Post-Training Pruning | 98.43%   | 95.61%   | 95.61%              |

The final pruned MobileNetV2 model achieved:

- **95.61%** sparsity (only 4.39% of weights remain non-zero)
- **85.34%** reduction in computational operations
- Only **97,747** non-zero parameters out of **2,224,736** total parameters
- Minimal accuracy drop of **0.93%** from the base model

These results demonstrate that SABR is effective even on architectures that are already optimized for mobile deployment, like MobileNetV2.

## Computing Operations for Sparse Networks

### Operations Calculation Formula

The computational cost of neural networks is typically measured in FLOPs (Floating Point Operations). For sparse models, we can calculate the reduced computational cost as follows:

#### For Convolutional Layers:

```
MACs = OutChannels * (InChannels/Groups) * KernelHeight * KernelWidth * OutputHeight * OutputWidth
Pruned MACs = MACs * (1 - SparsityRatio)
```

#### For Fully Connected (Linear) Layers:

```
MACs = OutputFeatures * InputFeatures
Pruned MACs = MACs * (1 - SparsityRatio)
```

Where:

- `MACs` are Multiply-Accumulate operations, the primary computation in neural networks
- `SparsityRatio` is the percentage of weights that are zero (e.g., 0.95 for 95% sparsity)

This calculation allows us to quantify the actual computational savings achieved through model pruning, which is crucial for understanding the efficiency gains for edge deployment.

## Leveraging Sparsity: Hardware and Compiler Solutions

While our results show impressive sparsity levels, standard deep learning frameworks and general-purpose hardware cannot fully leverage these benefits due to several challenges:

### Challenges in Utilizing Sparse Models

1. **Dense Storage Formats:** Standard frameworks store models in dense formats, occupying the same memory regardless of zero values.

2. **Memory Access Patterns:** Random sparsity patterns can lead to inefficient memory access due to cache misses and irregular memory fetches.

3. **SIMD Processing:** Modern CPUs and GPUs use SIMD (Single Instruction Multiple Data) units optimized for dense operations, not sparse computations.

4. **Matrix Multiplication Libraries:** Standard BLAS implementations are optimized for dense matrix operations, not sparse ones.

### Specialized Hardware Solutions

To fully realize the benefits of sparse models, specialized hardware architectures are necessary:

1. **Eyeriss:** A reconfigurable DNN accelerator that skips computations involving zero weights or activations (Chen et al., 2016).

2. **Sparse CNN Accelerator (SCNN):** Designed specifically for sparse CNNs, achieving 2.7x higher throughput compared to dense accelerators (Parashar et al., 2017).

3. **Cambricon-S:** A dedicated sparse neural network accelerator that achieves 7.23× speedup and 6.43× energy saving compared to dense implementation (Zhang et al., 2016).

4. **EIE (Efficient Inference Engine):** Hardware accelerator designed for compressed neural networks, delivering 13x speedup over dense GPU implementations (Han et al., 2016).

### Deep Learning Compiler Support

Modern deep learning compilers can generate optimized code for sparse operations:

1. **TVM (Tensor Virtual Machine):** Provides sparse tensor abstractions and optimizations that can target specific hardware (Chen et al., 2018).

2. **MLIR (Multi-Level Intermediate Representation):** Offers a unified framework for representing sparse tensor operations.

3. **Glow:** Facebook's compiler for neural network hardware accelerators with sparse tensor optimizations.

4. **XLA (Accelerated Linear Algebra):** Google's compiler for machine learning with support for sparse operations.

These specialized solutions can provide orders of magnitude improvement in performance and energy efficiency when deploying sparse models on edge devices.


## Conclusion

The SABR-Pruning project demonstrates the effectiveness of adaptive regularization for neural network pruning. By achieving over 97% sparsity with minimal accuracy loss, we show that even complex deep learning models can be made suitable for resource-constrained edge devices.

Our approach provides several advantages:

1. **Balanced Sparsity-Accuracy Tradeoff:** The adaptive regularization ensures that sparsity is induced without significant performance degradation
2. **Architecture-Agnostic:** Successfully applied to both VGG11 and MobileNetV2, showing versatility across model architectures
3. **Practical Application:** The real-time hand gesture recognition demo proves the practicality of the pruned models
4. **Significant Resource Reduction:** Over 93% reduction in computational operations makes deployment feasible on devices with limited resources

While standard frameworks don't fully utilize these benefits, specialized hardware accelerators and compilers are emerging that can exploit model sparsity for significant improvements in speed and energy efficiency. This project serves as a foundation for deploying complex AI models on edge devices, opening new possibilities for applications in IoT, mobile computing, and embedded systems.

## References

1. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both Weights and Connections for Efficient Neural Networks. Advances in Neural Information Processing Systems.

2. Chen, Y. H., Emer, J., & Sze, V. (2016). Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks. International Symposium on Computer Architecture (ISCA).

3. Parashar, A., Rhu, M., Mukkara, A., Puglielli, A., Venkatesan, R., Khailany, B., Emer, J., Keckler, S. W., & Dally, W. J. (2017). SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks. International Symposium on Computer Architecture (ISCA).

4. Zhang, S., Du, Z., Zhang, L., Lan, H., Liu, S., Li, L., Guo, Q., Chen, T., & Chen, Y. (2016). Cambricon-X: An Accelerator for Sparse Neural Networks. International Symposium on Microarchitecture (MICRO).

5. Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., Cowan, M., Wang, L., Hu, Y., Ceze, L., Guestrin, C., & Krishnamurthy, A. (2018). TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. Operating Systems Design and Implementation (OSDI).

6. Han, S., Mao, H., & Dally, W. J. (2016). Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. International Conference on Learning Representations (ICLR).

7. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. International Conference on Learning Representations (ICLR).
