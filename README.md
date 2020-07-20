# My reading list

## Security

### Federated learning

#### Byzantine attack & defence

- [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/pdf/1802.07927.pdf)

#### Inference attack & defence

### Backdoor
#### Attack
- [Hidden Trigger Backdoor Attacks](https://arxiv.org/pdf/1905.11468.pdf)
Proposed a backdoor attack which can make the poisoned image looks natural and is the same as the image from the target class, however, it is close to the source image with triger in the feature domain, which is evaluated by the ouptu of a certain layer. The optimization process mainly focus on two formulas, one is to shorten the L2 distance of poisoned image and the source image with trigger, the other is to ensure the poisoned image has the similar apperance to the images of target class.


#### Defence
- [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/pdf/1711.09404.pdf)
Training with gradient regularization increases robutness to adversarial perturbations, it can be combined with adversarial training to achieve greater robustness. Gradient Regularization actually adds another item, the rate of change of the energy with respect to the input features, to the loss formula except the common cross-entorpy, which can be seen as the energy of the network. The gradient regularization also increases the interpretability of adversarial pertubations.

- [Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization](https://arxiv.org/pdf/1601.07213v1.pdf)
Introduced DataGrad framework, which is framewrok that consider the layer-wise loss and regularization, the weights in this framework are updated by the gradient of the loss function and the loss of regularization, based on this model, many prior works of adversarial defecne can be explained, because most of their optimize objective can be written as the form of the equation proposed in this paper, and some of the works has the similar optimize objective as this equation.

- [Scaleable Input Gradient Regularization for Adversarial Roubutness](https://arxiv.org/pdf/1905.11468.pdf)

### Multi-Party Computation

### SGX and TrustZone

## Machine Learning

### Model Compression

#### Pruning

### Semantics Segmentation

