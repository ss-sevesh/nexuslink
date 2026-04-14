---
title: "Attention Is All You Need"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"]
doi: ""
arxiv_id: "1706.03762"
domain: ["cs.CL", "cs.LG"]
year: 2017
tags: []
---

## Summary

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## Key Findings

- We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
- Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU.
- On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
- We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## Methods

- [[Multi-Head Attention]]
- [[Feed-Forward Network]]
- [[Attention Mechanism]]
- [[Transformer]]
- [[Self-Attention]]
- [[Long Short-Term Memory]]
- [[Encoder-Decoder]]
- [[Recurrent Neural Network]]
- [[Layer normalization]]
- [[Deep learning]]
- [[Machine Learning]]

## Entities

- [[WMT]] (phenomenon)<!-- cx: On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the trai… -->
- [[BLEU]] (phenomenon)<!-- cx: On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the trai… -->
- [[ConvS2S]] (phenomenon)<!-- cx: In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for… -->
- [[Multi-Head Attention]] (method)<!-- cx: Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. -->
- [[Dot-Product Attention]] (phenomenon)<!-- cx: br>**----- End of picture text -----**<br>


**==> picture [122 x 186] intentionally omitted <==**

Figure 2: (left) Scaled Dot-Product Attention. -->
- [[Feed-Forward Network]] (method)<!-- cx: The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. -->
- [[RNN]] (phenomenon)<!-- cx: Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37]. -->
- [[WSJ]] (phenomenon)<!-- cx: We trained a 4-layer transformer with _dmodel_ = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank -->
- [[Attention Mechanism]] (method)<!-- cx: The best performing models also connect the encoder and decoder through an attention mechanism. -->
- [[Transformer]] (method)<!-- cx: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. -->
- [[Self-Attention]] (method)<!-- cx: Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. -->
- [[Long Short-Term Memory]] (method)<!-- cx: 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. Recurrent neural networks, long short-term memory -->
- [[Encoder-Decoder]] (method)<!-- cx: Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures -->
- [[Recurrent Neural Network]] (method)<!-- cx: Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Re… -->
- [[Layer normalization]] (method)<!-- cx: Layer normalization. -->
- [[Deep learning]] (method)<!-- cx: Deep learning with depthwise separable convolutions. -->
- [[Machine Learning]] (method)<!-- cx: Journal of Machine Learning Research_ , 15(1):1929–1958, 2014. -->

## References

