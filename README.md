# My reading list

## Security

### Federated learning

#### Byzantine attack & defence

- [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/pdf/1802.07927.pdf)

#### Inference attack & defence

- [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning](https://arxiv.org/pdf/1702.07464.pdf)


- [Deep Leakage from Gradients](https://arxiv.org/pdf/1906.08935.pdf)


### Backdoor & Adversarial
#### Attack
- [Hidden Trigger Backdoor Attacks](https://arxiv.org/pdf/1905.11468.pdf)
Proposed a backdoor attack which can make the poisoned image looks natural and is the same as the image from the target class, however, it is close to the source image with triger in the feature domain, which is evaluated by the ouptu of a certain layer. The optimization process mainly focus on two formulas, one is to shorten the L2 distance of poisoned image and the source image with trigger, the other is to ensure the poisoned image has the similar apperance to the images of target class.


#### Defence
- [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/pdf/1711.09404.pdf)
Training with gradient regularization increases robutness to adversarial perturbations, it can be combined with adversarial training to achieve greater robustness. Gradient Regularization actually adds another item, the rate of change of the energy with respect to the input features, to the loss formula except the common cross-entorpy, which can be seen as the energy of the network. The gradient regularization also increases the interpretability of adversarial pertubations.

- [Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization](https://arxiv.org/pdf/1601.07213v1.pdf)
Introduced DataGrad framework, which is framewrok that consider the layer-wise loss and regularization, the weights in this framework are updated by the gradient of the loss function and the loss of regularization, based on this model, many prior works of adversarial defecne can be explained, because most of their optimize objective can be written as the form of the equation proposed in this paper, and some of the works has the similar optimize objective as this equation.

- [Scaleable Input Gradient Regularization for Adversarial Roubutness](https://arxiv.org/pdf/1905.11468.pdf)

- [Towards Robust Training of Neural Networks by Regularizing Adversarial Gradients](https://arxiv.org/pdf/1712.00673.pdf)
Proposed a novel training method to defend the adversarial attack by designing a new regularizer function. The regularizer calculates the the difference between the maximum wrong logit output and the correct output, which can be considered as the fastest direction of pushing the the wrong logit to be larger than the correct logit output. The regularizer is added to the global loss function after derivating, and it need to be obtained by double-backpropagation. It is worth mentioning that the regularizer is calculated by logits that haven't passed throught the softmax layer or cross-entropy, which contains more direct information of the gradients. This improvement can make the edge of the model more smooth, so that it is hard for the adversarial attack to utilize the gradients in the similar magnitude to generate adversarial examples.

- [One Neuron to Fool Them All](https://arxiv.org/pdf/2003.09372.pdf)



### Multi-Party Computation

### SGX and TrustZone

## Machine Learning

### Model Compression

#### Pruning

### Semantics Segmentation

