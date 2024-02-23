---
layout: page
title: Proposal
permalink: /proposal/
---

## Introduction/Background

### Introduction
Facial expression classification has been an important problem in ML and CV ever since the inception of both fields. Recognizing facial emotions has always been a subjective, time-consuming, and strictly human task. Well-designed ML models that can predict facial expressions, emotions, and attitudes from just a snapshot of a face have the potential to optimize processes in the realms of human-computer interaction, psychology, security, and marketing.

### Literature Review
Artificial neural networks have been trained on data extracted from the Facial Action Coding System (FACS) to classify facial expressions based on the individual movements of specific muscle groups [1]. FACS extracts muscle movements from image headshots, which are then fed into the NN. Others have directly fed facial images into CNNs after facial detection and background removal. There have also been comparisons between classical machine learning models compared to newer deep learning models [7].

### Benchmarks

| Dataset | Best Performing Model | Accuracy |
| ------- | --------------------- | -------- |
| AffectNet [3] | DDAMFN [4] | 67.03% |
| RAF_DB [6]    | PAtt-Lite  | 95.05% |
