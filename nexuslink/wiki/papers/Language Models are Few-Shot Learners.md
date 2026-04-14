---
title: "Language Models are Few-Shot Learners"
authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss", "Gretchen Krueger", "Tom Henighan", "Rewon Child", "Aditya Ramesh", "Daniel M. Ziegler", "Jeffrey Wu", "Clemens Winter", "Christopher Hesse", "Mark Chen", "Eric Sigler", "Mateusz Litwin", "Scott Gray", "Benjamin Chess", "Jack Clark", "Christopher Berner", "Sam McCandlish", "Alec Radford", "Ilya Sutskever", "Dario Amodei"]
doi: ""
arxiv_id: "2005.14165"
domain: ["cs.CL"]
year: 2020
tags: []
---

## Summary

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

## Key Findings

- Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.
- Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting.
- Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans.

## Methods

- [[Pre-Training]]
- [[Fine-Tuning]]
- [[Few-Shot Learning]]
- [[Transformer]]
- [[In-Context Learning]]

## Entities

- [[NLP]] (phenomenon)<!-- cx: Jeffrey Wu Clemens Winter** 

**Christopher Hesse Mark Chen** 

**Eric Sigler Mateusz Litwin Scott Gray** 

**Benjamin Chess** 

**Jack Clark Christopher Berner** 

**Sam McCandlish Alec Radford** 

*… -->
- [[GPT-3]] (phenomenon)<!-- cx: Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. -->
- [[NLI]] (phenomenon)<!-- cx: .|18|
||3.8<br>NLI . . . . . . . . . . . . . . . . . . . . . . . . -->
- [[ANLI]] (phenomenon)<!-- cx: This includes natural language inference tasks like the ANLI dataset, and some reading comprehension datasets like RACE or QuAC. -->
- [[Sparse Transformer]] (phenomenon)<!-- cx: We use the same model and architecture as GPT-2 [RWC[+] 19], including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alt… -->
- [[GPU]] (phenomenon)<!-- cx: The precise architectural parameters for each model are chosen based on computational efficiency and load-balancing in the layout of models across GPU’s. -->
- [[Common Crawl]] (phenomenon)<!-- cx: ## **2.2 Training Dataset** 

Datasets for language models have rapidly expanded, culminating in the Common Crawl dataset[2] -->
- [[LAMBADA]] (phenomenon)<!-- cx: For LAMBADA and Storycloze there is no supervised training set available so we draw conditioning examples from the development set and evaluate on the test set. -->
- [[ARC]] (phenomenon)<!-- cx: For most tasks we compare the per-token likelihood (to normalize for length), however on a small number of datasets (ARC, OpenBookQA, and RACE) we gain additional benefit as measured on the developmen… -->
- [[BLEU]] (phenomenon)<!-- cx: We score the model using F1 similarity score, BLEU, or exact match, depending on what is standard for the dataset at hand. -->
- [[PTB]] (phenomenon)<!-- cx: |Setting|PTB|
|---|---|
|SOTA (Zero-Shot)|35.8_a_|
|GPT-3 Zero-Shot|**20.5**|



**Table 3.1: Zero-shot results on PTB language modeling dataset. -->
- [[ALUM]] (phenomenon)<!-- cx: GPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot setting, outperforming the 75.4% accuracy of a fine-tuned 1.5B parameter language model [ZHR[+] 19] but still a… -->
- [[BERT]] (phenomenon)<!-- cx: This is still 4.1% lower than the fine-tuned SOTA using a BERT based model [LDL19] but improves over previous zero-shot results by roughly 10%. -->
- [[Q&A]] (phenomenon)<!-- cx: Note that in addition to all results being in the closed-book setting, our use of few-shot, one-shot, and zero-shot evaluations represent an even stricter setting than previous closed-book QA work: in… -->
- [[T5 11B+SSM]] (phenomenon)<!-- cx: On Natural Questions (NQs) GPT-3 achieves 14.6% in the zero-shot setting, 23.0% in the one-shot setting, and 29.9% in the few-shot setting, compared to 36.6% for fine-tuned T5 11B+SSM. -->
- [[XLM]] (phenomenon)<!-- cx: En datasets as measured by `multi-bleu.perl` with XLM’s tokenization in order to compare most closely with prior unsupervised NMT work. -->
- [[BPE]] (phenomenon)<!-- cx: This could be a weakness due to reusing the byte-level BPE tokenizer of GPT-2 which was developed for an almost entirely English training dataset. -->
- [[Winograd-Style Task]] (phenomenon)<!-- cx: This is shown in Figure 3.4 in the case of few-shot results, and scaling for all three settings is shown in Appendix H. 

## **3.4 Winograd-Style Tasks** 

The Winograd Schemas Challenge [LDM12] is a… -->
- [[PIQA]] (phenomenon)<!-- cx: Recently fine-tuned language models have achieved near-human performance on the original Winograd dataset, but more difficult versions 

16 

|Setting|PIQA|ARC (Easy)|ARC (Challenge)|OpenBookQA|
|---|… -->
- [[GPT-3 Few-Shot PIQA result]] (phenomenon)<!-- cx: GPT-3 Few-Shot PIQA result is evaluated on the test server. -->
- [[WSC]] (phenomenon)<!-- cx: Note that this setting differs slightly from the WSC task in the SuperGLUE benchmark, which is presented as binary classification and requires entity extraction to convert to the form described in thi… -->
- [[Winograd GPT-3]] (phenomenon)<!-- cx: On Winograd GPT-3 achieves 88.3%, 89.7%, and 88.6% in the zero-shot, one-shot, and few-shot settings, showing no clear in-context learning but in all cases achieving strong results just a few points b… -->
- [[Winogrande]] (material)<!-- cx: On the more difficult Winogrande dataset, we do find gains to in-context learning: GPT-3 achieves 70.2% in the zero-shot setting, 73.2% in the one-shot setting, and 77.7% in the few-shot setting. -->
- [[BERT++]] (phenomenon)<!-- cx: The BERT-Large reference model was fine-tuned on the SuperGLUE training set (125K examples), whereas BERT++ was first fine-tuned on MultiNLI (392K examples) and SWAG (113K examples) before further fin… -->
- [[RTE]] (phenomenon)<!-- cx: On BoolQ, MultiRC, and RTE, performance is reasonable, roughly matching that of a fine-tuned BERT-Large. -->
- [[ANLI Round 3]] (phenomenon)<!-- cx: [376 x 229] intentionally omitted <==**

**Figure 3.9: Performance of GPT-3 on ANLI Round 3. -->
- [[BERT Large]] (phenomenon)<!-- cx: On RTE, only the largest version of GPT-3 performs convincingly better than random (56%) in any evaluation setting, but in a few-shot setting GPT-3 performs similarly to a single-task fine-tuned BERT… -->
- [[SAT]] (phenomenon)<!-- cx: Third, we test GPT-3’s ability to solve SAT-style analogy problems few-shot. -->
- [[Student’s T-Test]] (phenomenon)<!-- cx: We provide one to five previous examples of a (separate) 

> 5We use a two-sample Student’s T-Test to test for significant difference between the means of the participant accuracies of each model and… -->
- [[Pre-Training]] (method)<!-- cx: Jeffrey Wu Clemens Winter** 

**Christopher Hesse Mark Chen** 

**Eric Sigler Mateusz Litwin Scott Gray** 

**Benjamin Chess** 

**Jack Clark Christopher Berner** 

**Sam McCandlish Alec Radford** 

*… -->
- [[Fine-Tuning]] (method)<!-- cx: Jeffrey Wu Clemens Winter** 

**Christopher Hesse Mark Chen** 

**Eric Sigler Mateusz Litwin Scott Gray** 

**Benjamin Chess** 

**Jack Clark Christopher Berner** 

**Sam McCandlish Alec Radford** 

*… -->
- [[Few-Shot Learning]] (method)<!-- cx: At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corp… -->
- [[Transformer]] (method)<!-- cx: First, single-layer representations were learned using word vectors [MCCD13, PSM14] and fed to task-specific architectures, then RNNs with multiple layers of representations and contextual state were… -->
- [[In-Context Learning]] (method)<!-- cx: We use the term “in-context learning” to describe the inner loop of this process, which occurs within the forward-pass upon each sequence. -->
- [[Entropy]] (phenomenon)<!-- cx: One might worry that these improvements in cross-entropy loss come only from modeling spurious details of our training corpus. -->

## References

