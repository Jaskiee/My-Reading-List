# My reading list

## Security

### Federated learning

#### Byzantine attack & defence

- [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/pdf/1802.07927.pdf)

#### Inference attack & defence

### Backdoor
#### Attack

#### Defence
- [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/pdf/1711.09404.pdf)
Training with gradient regularization increases robutness to adversarial perturbations, it can be combined with adversarial training to achieve greater robustness. Gradient Regularization actually adds another item, the rate of change of the energy with respect to the input features, to the loss formula except the common cross-entorpy, which can be seen as the energy of the network. The gradient regularization also increases the interpretability of adversarial pertubations.

- [Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization](https://arxiv.org/pdf/1601.07213v1.pdf)
Introduce DataGrad framework, which is framewrok that consider the layer-wise loss and regularization, the weights in this framework are updated by the gradient of the loss function and the loss of regularization, based on this model, many prior works of adversarial defecne can be explained, because most of their optimize objective can be written as the form of the equation proposed in this paper, and some of the works has the similar optimize objective as this equation.


### Multi-Party Computation

### SGX and TrustZone

## Machine Learning

### Model Compression

#### Pruning

### Semantics Segmentation

