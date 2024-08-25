---
layout: post
title: "Text Generation Interface (TGI) Review"
date: 2024-08-23 14:00:00 +0900
categories: IT
author: dh.ihm
---

[TGI Document](https://huggingface.co/docs/text-generation-inference/index)

TGI의 소개 페이지에서는 맨 처음 여러가지 최적화와 기능들을 구현했다고 말하고 있습니다. 

그 중에서 다음 몇가지 항목들에 대한 리뷰를 하고 정리해보겠습니다.

- [Tensor Parallelism for faster inference on multiple GPUs](TGI_review_1.md)
- Tokne streaming using Server-Senf Events (SSE)
- Continuous batching of incoming requests for increased total throughput
- Optimized transformers code for inference using Flash Attantion and Paged Attention on the most popular architectures
- Quantization with bitsandbytes and GPT-Q
- Stop sequences

그리고 server 및 client 코드를 살펴보고, 각각의 기능들에 대한 리뷰를 해보겠습니다. 

- [TGI Review - server](TGI_review_2.md)
- TGI Review - client

source: `{{ page.path }}`
