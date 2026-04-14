---
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
authors: ["William Fedus", "Barret Zoph", "Noam Shazeer"]
doi: ""
arxiv_id: "2101.03961"
domain: ["cs.LG", "cs.AI"]
year: 2021
tags: []
---

## Summary

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.

## Key Findings

- Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats.
- We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources.
- Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.

## Methods

- [[Feed-Forward Network]]
- [[Machine Learning]]
- [[Deep Learning]]
- [[Mixture of Expert]]
- [[Transformer]]
- [[Pre-Training]]
- [[Fine-Tuning]]
- [[Self-Attention]]

## Entities

- [[MoE]] (material)<!-- cx: However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs, and training instability. -->
- [[Switch Transformer]] (phenomenon)<!-- cx: We address these with the introduction of the Switch Transformer. -->
- [[T5-Large]] (phenomenon)<!-- cx: We design models based off T5-Base and T5-Large (Raffel et al., 2019) to obtain up to 7x increases in pre-training speed with the same computational resources. -->
- [[mT5]] (phenomenon)<!-- cx: These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. -->
- [[NLP]] (phenomenon)<!-- cx: We measure superior scaling on a diverse set of natural language tasks and across three regimes in NLP: pre-training, finetuning and multi-task training. -->
- [[T5-XXL]] (material)<!-- cx: These models improve the pre-training speed of a strongly tuned T5-XXL baseline by 4x. -->
- [[TPU]] (phenomenon)<!-- cx: Our work here focuses on TPU architectures, but these class of models may be similarly trained on GPU clusters. -->
- [[GPU]] (phenomenon)<!-- cx: Our work here focuses on TPU architectures, but these class of models may be similarly trained on GPU clusters. -->
- [[FFN Layer]] (phenomenon)<!-- cx: [390 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
y1 y2<br>Add + Normalize<br>y<br>FFN 1 FFN 2 FFN 3 FFN 4 FFN 1 FFN 2 FFN 3 FFN 4<br>Add + Normalize<br>p = 0.8<br>Swi… -->
- [[FFN]] (phenomenon)<!-- cx: We replace the dense feed forward network (FFN) layer present in the Transformer with a sparse Switch FFN layer (light blue). -->
- [[MoE Transformer]] (material)<!-- cx: We highlight three key findings from Table 1: **(1)** Switch Transformers outperform both carefully tuned dense models and MoE Transformers on a speed-quality basis. -->
- [[FLOPS]] (phenomenon)<!-- cx: and thus its FLOPS are larger. -->
- [[GLUE]] (phenomenon)<!-- cx: The language benchmarks GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019) are handled as composite mixtures with all the tasks blended in proportion to the amount of tokens present in each. -->
- [[MRPC]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[STS]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[QQP]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[MNLI]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[RTE]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[WNLI]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[WSC]] (phenomenon)<!-- cx: These benchmarks consist of tasks requiring sentiment analysis (SST2), word sense disambiguation (WIC), sentence similarty (MRPC, STS-B, QQP), natural language inference (MNLI, QNLI, RTE, CB), questio… -->
- [[BBC XSum]] (phenomenon)<!-- cx: The CNNDM (Hermann et al., 2015) and BBC XSum (Narayan et al., 2018) data sets are used to measure the ability to summarize articles. -->
- [[ARC Reasoning Challenge]] (phenomenon)<!-- cx: Question answering is probed with the SQuAD data set (Rajpurkar et al., 2016) and the ARC Reasoning Challenge (Clark et al., 2018). -->
- [[CNNDM]] (phenomenon)<!-- cx: The Rouge-2 metric is used both the CNNDM and XSum. -->
- [[ARC Challenge]] (phenomenon)<!-- cx: Finally, in ARC Easy, ARC Challenge, ANLI, and Winogrande we report the accuracy of the generated responses. -->
- [[ANLI]] (phenomenon)<!-- cx: Finally, in ARC Easy, ARC Challenge, ANLI, and Winogrande we report the accuracy of the generated responses. -->
- [[AI2 Reasoning Challenge]] (phenomenon)<!-- cx: In our fine-tuning study, the only tasks where we do not observe gains are on the AI2 Reasoning Challenge (ARC) data sets where the T5-Base outperforms Switch-Base on the challenge data set and T5-Lar… -->
- [[ARC]] (phenomenon)<!-- cx: In our fine-tuning study, the only tasks where we do not observe gains are on the AI2 Reasoning Challenge (ARC) data sets where the T5-Base outperforms Switch-Base on the challenge data set and T5-Lar… -->
- [[BERT]] (phenomenon)<!-- cx: These techniques are built off of Sanh et al. (2019), who study distillation methods for BERT models. -->
- [[T5-Base]] (phenomenon)<!-- cx: For a final baseline, we find no improvement of T5-Base initialized with the expert weights, but trained normally without distillation. -->
- [[Common Crawl]] (phenomenon)<!-- cx: We pre-train on the multilingual variant of the Common Crawl data set (mC4) spanning 101 languages introduced in mT5, but due to script variants within certain languages, the mixture contains 107 task… -->
- [[mC4]] (phenomenon)<!-- cx: We pre-train on the multilingual variant of the Common Crawl data set (mC4) spanning 101 languages introduced in mT5, but due to script variants within certain languages, the mixture contains 107 task… -->
- [[mT5-Base]] (phenomenon)<!-- cx: In Figure 7 we plot the quality improvement in negative log perplexity for all languages of a FLOP-matched Switch model, mSwitch-Base to the T5 base variant, mT5-Base. -->
- [[SPMD]] (phenomenon)<!-- cx: Once it exceeds the size of the accelerator’s memory, single program multiple data (SPMD) modelparallelism can be employed. -->
- [[Feed-Forward Network]] (method)<!-- cx: **Reviewing the Feed-Forward Network (FFN) Layer. -->
- [[Mesh TensorFlow]] (phenomenon)<!-- cx: We use the FFN layer as an example of how data, model and expert-parallelism works in Mesh TensorFlow (Shazeer et al., 2018) and review it briefly here. -->
- [[GPT-3]] (phenomenon)<!-- cx: It is common to mix both model and data parallelism for large scale models, which was done in the largest T5 models (Raffel et al., 2019; Xue et al., 2020) and in GPT-3 (Brown et al., 2020). -->
- [[FFNGEGLU]] (phenomenon)<!-- cx: Standard hyper-parameters of the Transformer, including _dmodel_ , _dff_ , _dkv_ , number of heads and number of layers are described, as well as a less common feature, _FFNGEGLU_ , which refers to a… -->
- [[XLA]] (phenomenon)<!-- cx: More recently, through advances in machine learning infrastructure, GShard (Lepikhin et al., 2020), which extended the XLA compiler, used the MoE Transformer to dramatically improve machine translatio… -->
- [[SSM]] (phenomenon)<!-- cx: We correlate the upstream performance with downstream quality on both SuperGLUE and TriviaQA (SOTA recorded without SSM), reasoning and knowledge-heavy benchmarks, respectively (validation sets). -->
- [[Machine Learning]] (method)<!-- cx: Journal of Machine Learning Research 23 (2022) 1-40 

Submitted 8/21; Revised 3/22; Published 4/22 

# **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**… -->
- [[Deep Learning]] (method)<!-- cx: Alexander Clark 

## **Abstract** 

In deep learning, models typically reuse the same parameters for all inputs. -->
- [[Mixture of Expert]] (method)<!-- cx: Mixture of Experts (MoE) models defy this and instead select _different_ parameters for each incoming example. -->
- [[Transformer]] (method)<!-- cx: We address these with the introduction of the Switch Transformer. -->
- [[Pre-Training]] (method)<!-- cx: We design models based off T5-Base and T5-Large (Raffel et al., 2019) to obtain up to 7x increases in pre-training speed with the same computational resources. -->
- [[Fine-Tuning]] (method)<!-- cx: ||2.4<br>Improved Training and Fine-Tuning Techniques . . . . . . . . . . . . . . . -->
- [[Self-Attention]] (method)<!-- cx: [390 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
y1 y2<br>Add + Normalize<br>y<br>FFN 1 FFN 2 FFN 3 FFN 4 FFN 1 FFN 2 FFN 3 FFN 4<br>Add + Normalize<br>p = 0.8<br>Swi… -->
- [[Entropy]] (phenomenon)<!-- cx: [−]_[2] which was sufficiently large to ensure load balancing while small enough to not to overwhelm the primary cross-entropy objective. -->

## References

