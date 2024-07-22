# Adaptive Retrieval-based Gradient Planning for Offline Multi-context Model-based Optimization
A `PyTorch` implementation of our paper.

If you have any questions, please contact the author: [Haoran Zhao](http://apex.sjtu.edu.cn/members/cmc_iris@apexlab.org).

## Abstract
> Offline model-based optimization aims to find designs that maximize a black-box function with only offline datasets. Under this setting, based on different practical application scenarios, many variant settings have been raised. For example, dimensional constrained MBO discusses situations where the achievable design space is constrained by environmental factors, thus requiring certain dimensions to maintain constant values during the optimization process. We find that it is the diverse contextual data situations that lead to different state-of-the-art methods being applicable to these settings respectively. Even within the optimization process of one setting, contextual data situations can also vary significantly, resulting in inconsistent performance across different datasets. To address this challenge, in this paper we define the problem of offline multi-context MBO, which takes different contextual data situations into consideration, and propose an innovative framework named Adaptive Retrieval-based gradient Planning for Optimization (ARPO), which leverages different retrieval strategies, including naive retrieval, gradient-direction retrieval and average-direction retrieval, to generate different gradient directions for various data contexts. ARPO is able to plan the gradients adaptively according to the evaluations of the retrieval model, since the model, as we will show, exhibits high predictive accuracy and implicitly introduces fine-grained conservatism against out-of-distribution designs. Experiments are conducted on the Hartmann test function and the Design-Bench benchmark, where ARPO outperforms the state-of-the-art methods in multi-context MBO tasks.
