# uncertainty-estimates


# Objective

A project to find out a **practical approach** for quantifying the uncertainty of the model predictions.(in deep neural networks). Thus, training an auxillary generative model doesn't make the cut as it introduces a completely new problem.(accidental complexity). The objective is to strive to find a simple approach,that can be widely used in the industry. This is crucial for deploying many ML/NN based models in **high-stakes environments** as we should know when ***we can't trust the model outputs.**


## Backgroud

Naively using softmax probabilities to detect/quantify is wrong. They do not work. This has already been shown. Typically when fed unseen/out-of-sample data-point, softmax probabilities will still be high and would be statistically undecipherable from other insample points. 

### Fate of naively using softmax outputs
Below plot shows the histogram of the model(Resnet18)-entropies trained on MNIST on three data-groups - **unseen MNIST samples**,**FMNIST samples**,**Some ambiguos MNIST samples**(these are generated via an VAE ,esentially images that cannot be put into a unique class,like a digit that looks both like a 3 and 9.).

<img width="1050" alt="Screen Shot 2022-01-21 at 2 01 46 PM" src="https://user-images.githubusercontent.com/21222766/150585087-76fe4392-217d-45d6-afaa-4e1bb304da01.png">

I will spare further details as this is a work in progress but there exits a simple approach to not only identidy model uncertainty but also disambiguate between alaetoric and epistemic uncertainties. Check the below plot, on same samples, using a simple method, we can seperate between *same-distribution but oos* vs *different distribution* points.

<img width="1026" alt="Screen Shot 2022-01-21 at 2 01 31 PM" src="https://user-images.githubusercontent.com/21222766/150587919-3ebfcfd4-3546-4ca7-bbac-1c31d460bab6.png">



**Research scratch pad**.(**work in progress**)


 Below is the notes to myself as i explore this problem.
## Step-1: Literature Survey
ref1: [Safer Classification by Synthesis](https://arxiv.org/abs/1711.08534)

`Synthesis`

- The `probability values` of Discriminative classifiers can be high even when the input is coming from out-of-sample distribution. This can be quantified by plotting `coverage` vs `risk` (by varying the classification threshold). Ideally when sufficiently large threshold is chosen, misclassification-rate should be zero.
- **IDEA:Generative Classification** Since discrimative classification can be seen as finding the projecting of the data onto a manifold such that seperating hyperplanes neatly segregate the data. Now, we can see how even for an out-of-distribution sample, we can have high-confidence prediction. Thus, We can try to model the `distance from the boundary` directly to quantify the confidence. 
- Let's say we have a class-conditional generative model  **M**. For a given test image `x`, find an image from each class that is closest(some distance-metric like l2) to it via the generative model, and label it's class to be the one that has highest similarity(least distance). Here we can use a threshold to determine if we can assign a class label or not. Here we might be able to avoid the below case.
- **Conclusion**: Use the generative model for thresholding. Use a CNN to predict the labels on these images. This gives us both the accuracy and shields us from making predictions on out-of-sample data-points.

 <img width="578" alt="Screen Shot 2021-12-27 at 3 29 17 PM" src="https://user-images.githubusercontent.com/21222766/147505241-a0cf3c76-dd6d-4fda-8563-ad2ccd4b9386.png">

`Implementation concerns`

- Monte-carlo sampling and L-BFGS is needed to find the optimal image for each class.

`Missing Pieces/Think on these things`

- Is the additional complexity warranted ? Can we use simpler approaches ?
- Generating images/samples is not needed for the task at hand. We can fit the generative model on a lower dimensional projection vectors.
- Decoupling of objectives is typically a bad idea in deep learning. Can we frame a sigle network/objective to perform both novelty detection and classification ? Note that the paper is quite old.
- Find some real-world data with `data-shift`,to ultimately bench mark your approach. Also think/quantify exactly what do you mean by `out-of-distribution` and `uncertainity`(epistemic vs aleatory uncertainty). Bayesian NN's seem to be a direct candidate here.
`Ideas`:
Since descriminative classification doesn't explictly have a distance metric, we can train networks with explicit [distance metric for classification](https://arxiv.org/abs/1703.05175) and use thresholding to detect out-of-sample distributions. Here's another [paper](https://arxiv.org/abs/2003.02037),[Deterministic Neural Networks with Inductive Biases Capture Epistemic and
Aleatoric Uncertainty](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-022.pdf),[Pitfalls of OOD detection](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-092.pdf) that does exactly this.

Ideally we would want to classify samples into a. Unseen out-of-sample(alaetoric) b. Too hard to detect into a class(epistemic uncertianty). This would play a big role in debugging and improving the models. Moreover Generative models aren't great at [identyfying OOD samples.](https://arxiv.org/abs/1810.09136),[this](https://arxiv.org/abs/1810.01392)

**NOTES**

`On Simpler Approaches`
- Train an ensemble of models(can be done within a single model with multiple heads) and use the `std` as a threshold to decide for `novelty` and the `avg` as the class label.

ref 2: [A survey on uncertainty estimation](https://arxiv.org/abs/2107.03342)

ref 3: [NeurIps Shifts Challenge](https://research.yandex.com/shifts/)

Here are some cool empirical studies:
Tabular data: [Paper](https://arxiv.org/abs/2112.03566) and [Code](https://github.com/bond005/yandex-shifts-weather)
Image data :[Paper](https://arxiv.org/abs/2112.04350)
Guidelines to tackle [realworld shifts](https://vectorinstitute.ai/wp-content/uploads/2021/08/ds_project_report_final_august9.pdf)

ref 3: [Collection of real-world datasets with distribution shift](http://ai.stanford.edu/blog/wilds/)

## Final Experimental Setup
DataSets: MNIST(training data),DIRTYMNIST(in-d but uncertain samples),FASHIONMNIST(to get ood samples)
Perform temperature scaling as done [here](https://openreview.net/pdf?id=BJxI5gHKDr) to compare the log-likelihood between the models.
Baseline: 1. [DeepEnsembles](https://arxiv.org/abs/1612.01474).(multiple models+possibly trained with adverserial training)
Above is SOTA interms of uncertainty quantification. But only bottleneck being 

1.Doesn't really differentiate between ambiguous inputs(alaetoric) and Clear out of sample inputs.
2. Training cost.

Can we have a simpler single model ?

Naively trained discriminative classifiers cannot do both classification and out-of-sample detection both efficiently. Reasons and a good first approach is given here:[Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108). It's single forward pass but a considerable departure from typical training. Here's a far simple approach - [code](https://github.com/omegafragger/DDU),[paper](https://arxiv.org/abs/2102.11582). Implement and compare with this model.





# Tools
We will be using the following tools for implementation.

[Pytorch-lightning](https://devblog.pytorchlightning.ai/lightning-tutorials-in-collaboration-with-the-university-of-amsterdam-uva-2499eaa0caad)

[Weights-Biases](https://wandb.ai/site)

[Streamlit](https://streamlit.io/)

For a good tutorial on deep learning and pytorch refer [Alex samola's book](https://d2l.ai/)
