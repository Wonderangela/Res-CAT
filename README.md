
To harness the potential of H&E image patches for gene expression prediction, our initial step involves leveraging established deep learning architectures such as ResNet50 for feature extraction. 

In our endeavor to analyze H&E images for spot interaction, we propose the development of a spot-interaction module. This module aims to refine predictions of gene expression. To streamline the model and minimize the number of parameters, we introduce a non-parametric attentive module. The operation of this attention module is defined by the following Equation:
A(Q,K,V)=softmax((QK^T)/√d)V
where K, Q, and V represent the key, query, and value matrices. To model the spot interaction, we aim to learn a compact gene expression correlation between the spots in the training set. Here, X_i represents the input features, and F(∙) and G(∙) denote transformation functions. The aim of this formulation is to explore the matrix of spot-spot
interactions. By doing so, it clusters spots that exhibit high correlation, thereby enhancing the robustness of the gene expression representation. Moreover, this approach elucidates the correlation among spots, improving the model’s inference capabilities.
To enhance the model’s learning process by integrating knowledge of ground-truth gene expressions, we introduce a secondary MSE loss function. This additional MSE loss is formulated as follows: 
L_mse^'=∑_(i=1)^N▒‖y_i-A(G(F(X_i )))‖_(l_2 ) ,
Integrating dual objectives into a unified framework, we define the overall loss function as follows:
L=〖L_mse+L〗_mse^'.
This combined loss framework is designed to enhance the model’s predictive performance. By carefully summing up the contribution of the primary and secondary MSE losses, the model is steered to pay detailed attention to the subtleties and complexities of gene expression data. This, in turn, is expected to improve the model’s capacity for capturing the intricate biological relationships that are represented within H&E images.


![image](https://github.com/Wonderangela/ResSAT/assets/51802393/74c8cfac-c4a9-4c47-947d-6a7f67b4fa9e)
