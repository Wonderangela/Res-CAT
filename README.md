## ResSAT: Enhancing Spatial Transcriptomics Prediction from H&E-Stained Histology Images with Interactive Spot Transformer

To harness the potential of H&E image patches for gene expression prediction, our initial step involves leveraging established deep learning architectures such as ResNet50 for feature extraction. 

In our endeavor to analyze H&E images for spot interaction, we propose the development of a spot-interaction module. This module aims to refine predictions of gene expression. To streamline the model and minimize the number of parameters, we introduce a non-parametric attentive module. To model the spot interaction, we aim to learn a compact gene expression correlation between the spots. By doing so, it clusters spots that exhibit high correlation, thereby enhancing the robustness of the gene expression representation. Moreover, this approach elucidates the correlation among spots, improving the model’s inference capabilities. This, in turn, is expected to improve the model’s capacity for capturing the intricate biological relationships that are represented within H&E images.


![image](https://github.com/Wonderangela/ResSAT/assets/51802393/74c8cfac-c4a9-4c47-947d-6a7f67b4fa9e)
