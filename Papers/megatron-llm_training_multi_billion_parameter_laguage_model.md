---
layout: post
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
date: 2024-09-04 19:02:00 +0900
categories: Papers
author: dh.ihm
---

이 연구에서는 수십억 개의 parameter를 갖는 transformer model을 training 할 수 있도록 해주는 간단하고 효율적인 model parallelism에 대한 내용을 구현 했습니다. 
논문에서는 새로운 컴파일러나, 라이브러리의 변경도 필요하지 않고, pipeline parallelism과 상호 보완적으로 동작할 수 있으며, PyTorch에 몇 가지 communication 작업을 삽입하는 것만으로도 완전히 구현 될 수 있다고 하고 있습니다. 

어떤 식으로 구현하였는지 한 번 살펴 보겠습니다. 
