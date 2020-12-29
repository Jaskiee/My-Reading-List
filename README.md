# My Reading list
This is a reading list of the papers that I am fond of, most of them are relating to deep learning, privacy-preserving machine learning and federated learning.


## Security

### Federated learning

#### Byzantine attack & defence

- [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/pdf/1802.07927.pdf)

- [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/pdf/1911.11815.pdf) (USENIX 2020)


#### Inference attack & defence

- [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning](https://arxiv.org/pdf/1702.07464.pdf)


- [Deep Leakage from Gradients](https://arxiv.org/pdf/1906.08935.pdf)

#### Backdoor

- [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://proceedings.neurips.cc/paper/2020/hash/b8ffa41d4e492f0fad2f13e29e1762eb-Abstract.html) (NeurIPS 2020) <br/>


### Adversarial

- [Improved Sample Complexities for Deep Neural Networks and Robust Classification via an All-Layer Margin](https://openreview.net/forum?id=HJe_yR4Fwr) (ICLR 2020)


- [High-frequency Component Helps Explain the Generalization of Convolutional Neural Networks](https://arxiv.org/pdf/1905.13545.pdf) (CVPR 2020)


- [One Neuron to Fool Them All](https://arxiv.org/pdf/2003.09372.pdf) (arXiv 2020)<br/> 
This work proposed the notion of sensitivity of individual neurons, which is to evaluate how roubst the model's output is to perturbations of the neuron's output.  A robust training method is also be proposed by adding a regularization term, to prevent the model giving high importance to some specific feature for a paticular class, and to ensure that no single feature has a high relative contribution to an input's corresponding logits.

- [Feature Denoising for Improving Adversarial Robustness](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Feature_Denoising_for_Improving_Adversarial_Robustness_CVPR_2019_paper.pdf) (CVPR 2019)


- [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175.pdf) (NeurIPS 2019) <br/> 
This paper attribute the robustness to some of the features which are highly predictive but incomprehensible to humans, and they also analyze that the adversarial vulnerability can be expressed as a difference between the inherent data metric and the l2 metric.


- [NIC: Detecting Adversarial Samples with Neural Network Invariant Checking](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf) (NDSS 2019)

- [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf) (ICLR 2014)<br/>
The first paper in adversarial examples.


#### Transferability
- [Improving Transferability of Adversarial Examples with Input Diversity](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.pdf) (CVPR 2019) <br/> 
Integrated Iterative Fast Gradient Sign Method (I-FGSM) with data augmentation, which can increase the diversity of the data. They transform the image randomly with the probability of p before each iteration in I-FGSM, the transformations include random resizing and random padding, this method can enhance the generalization of the examples so that increase their transferability.


#### Gradient Regularization

- [Scaleable Input Gradient Regularization for Adversarial Roubutness](https://arxiv.org/pdf/1905.11468.pdf)

- [Towards Robust Training of Neural Networks by Regularizing Adversarial Gradients](https://arxiv.org/pdf/1712.00673.pdf) <br/> 
They design a novel regularizer, the difference between the maximum wrong logit output and the correct output was considered as the fastest direction of pushing the the wrong logit to be larger than the correct logit output. This improvement can make the edge of the model more smooth, and harder for the adversarial attack to utilize the gradients to generate adversarial examples.

- [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/pdf/1711.09404.pdf) <br/> 
Training with gradient regularization can be combined with adversarial training to achieve greater robustness. Gradient Regularization actually adds the rate of change of the energy with respect to the input features to the loss formula besides common cross-entorpy, which can be seen as the energy of the network. The gradient regularization also increases the interpretability of adversarial pertubations.

- [Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization](https://arxiv.org/pdf/1601.07213v1.pdf) (arXiv 2016)<br/> 
Introduced DataGrad framework, which is a framewrok that consider the layer-wise loss and regularization, the weights in this framework are updated by the gradient of the loss function and the loss of regularization, and many prior works of adversarial defecne can be explained based on this model.

#### Lipchitz Constant

- [Parseval Networks: Improving Robustness to Adversarial Examples](https://arxiv.org/pdf/1704.08847.pdf)

- [L2-Nonexpansive Neural Networks](https://arxiv.org/pdf/1802.07896.pdf)

- [Understanding Adversarial Robustness: The Trade-Off Between Minimum and Average Margin](https://arxiv.org/pdf/1907.11780.pdf)


### Backdoor

#### Insight


- [On the Trade-off between Adversarial and Backdoor Robustness](https://papers.nips.cc/paper/2020/hash/8b4066554730ddfaa0266346bdc1b202-Abstract.html) (NeurIPS 2020) <br/>
This paper finds that there is a trade-off between adversarial and backdoor, models used adversarial training are more vulnerable to backdoor, and adversarial training impairs the effectiveness of the pre-training backdoor defense methods, while enhances the post-training defense method.


- [On Certifying Robustness against Backdoor Attacks via Randomized Smoothing](http://arxiv.org/abs/2002.11750) (CVPR 2020) <br/>

#### Attack

- [Reﬂection Backdoor: A Natural Backdoor Attack on Deep Neural Networks](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550188.pdf) (ECCV 2020) <br/>

- [Hidden Trigger Backdoor Attacks](https://arxiv.org/pdf/1905.11468.pdf) (AAAI 2020) <br/> 
Proposed a backdoor attack which can make the poisoned image looks natural and is the same as the image from the target class, however, it is close to the source image with triger in the feature domain, which is evaluated by the ouptu of a certain layer. They shorten the L2 distance of poisoned image and the source image with trigger, and ensure the poisoned image has the similar apperance to the images of target class.

- [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://proceedings.neurips.cc/paper/2018/hash/22722a343513ed45f14905eb07621686-Abstract.html) (NeurIPS 2018) <br/>



#### Defense
- [One-pixel Signature: Characterizing CNN Models for Backdoor Detection](http://arxiv.org/abs/2008.07711) (ECCV 2020) <br/>
Introduced a backdoor detection that generating a signature according to a neural network and determining that if there is a bookdoor in the network according to the signature. The signature is generated by a baseline input (with changing its value pixel by pixel), which can be seen as the sensitivity of the inside of the network to the input pixel.


- [Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kolouri_Universal_Litmus_Patterns_Revealing_Backdoor_Attacks_in_CNNs_CVPR_2020_paper.pdf) (CVPR 2020) <br/>
Proposed the Universal Litmus Patterns (ULPs), which can tell if the network is poisoned by feeding some universal patterns to the network and analyzing the output. The ULPs are trainable and they are gained by training on hundreds of poisoned and clean models.

- [ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation](https://dl.acm.org/doi/pdf/10.1145/3319535.3363216) (CCS 2019) <br/> 

- [Defending Neural Backdoors via Generative Distribution Modeling](https://papers.nips.cc/paper/2019/file/78211247db84d96acf4e00092a7fba80-Paper.pdf) (NeurIPS 2019) <br/>
This work used GAN and Max-Entropy Staircase Approximator (MESA) to generate all possible triggers, then retrained the model to remove the backdoor.

- [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html) (NeurIPS 2018) <br/> 

### Multi-Party Computation

- [SecureNN: 3-Party Secure Computation for Neural Network Training](https://eprint.iacr.org/2018/442.pdf)

- [SecureML: A System for Scalable Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/396.pdf)

- [ABY3: A Mixed Protocol Framework for Machine Learning](https://eprint.iacr.org/2018/403.pdf)


### SGX and TrustZone

- [SoK: General Purpose Compilers for Secure Multi-Party Computation](https://marsella.github.io/static/mpcsok.pdf)

- [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](https://arxiv.org/pdf/1806.03287.pdf)

## Machine Learning

### Model Compression

#### Pruning

### Semantics Segmentation

- [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf)

- [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)

- [ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1906.09826v1.pdf)

- [LEDNet: A Lightweight Encoder-Decoder Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1905.02423.pdf)

- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1808.00897.pdf)

- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/pdf/1704.08545.pdf)

- [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf)

- [ShuffleSeg: Real-time Semantic Segmentation Network](https://arxiv.org/pdf/1803.03816.pdf)

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf) (DeepLab v3+)

### Panoptic Segmentation

- [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf)

- [An End-to-End Network for Panoptic Segmentation](https://arxiv.org/pdf/1903.05027.pdf)

- [DeeperLab: Single-Shot Image Parser](https://arxiv.org/pdf/1902.05093.pdf)
