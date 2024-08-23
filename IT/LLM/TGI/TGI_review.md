---
layout: post
title: "ext Generation Interface (TGI) Review"
date: 2024-08-23 14:00:00 +0900
categories: IT
---

# Text Generation Interface (TGI) Review

[TGI](https://huggingface.co/docs/text-generation-inference/index)

TGI의 소개 페이지에서는 맨 처음 여러가지 최적화와 기능들을 구현했다고 말하고 있습니다. 
그 중에서 다음 몇가지 항목들에 대한 리뷰를 하고 정리해보겠습니다.

- [Tensor Parallelism for faster inference on multiple GPUs](TGI_review_1.md)
- Tokne streaming using Server-Senf Events (SSE)
- Continuous batching of incoming requests for increased total throughput
- Optimized transformers code for inference using Flash Attantion and Paged Attention on the most popular architectures
- Quantization with bitsandbytes and GPT-Q
- Stop sequences

source: `{{ page.path }}`
