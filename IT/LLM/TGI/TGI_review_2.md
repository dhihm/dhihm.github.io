---
layout: post
title: "TGI Review - server"
date: 2024-08-24 09:24:00 +0900
categories: IT
author: dh.ihm
---

[참고 자료: TGI Server - server.py](https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/server.py)

## TextGenerationService class

이 class는 text generatoin과 관련된 여러 gRPC method를 포함합니다. 

코드는 다음과 같습니다. 

```python
class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
```

generate_pb2_grpc.TextGenerationServiceServicer를 상속받아서 구현하고 있습니다. 

이름에서 유추해 볼 수 있듯이, generate_pb2_grpc.TextGenerationServiceServicer는 gRPC framework에서 자동으로 생성된 코드입니다. 

gRPC에 대한 내용은 간략하게만 설명하겠습니다. :stuck_out_tongue:

gRPC는 Google에서 개발한 RPC framework로, protobuf를 사용합니다. 

protobuf는 구조화된 데이터를 직렬화하고, 역직렬화하는데 사용되는 데이터 포맷입니다.

protobuf에 아래와 같이 service를 정의하면, gRPPC framework에서 이 service를 기반으로 service interface 코드를 자동으로 생성해 줍니다. 

```proto
service TextGenerationService {
  rpc MethodA(InputParam) returns (OutputResponse) {}
  rpc MethodB(GenerateTextsRequest) returns (GenerateTextsResponse) {}
  ...
}
``` 

따라서 generate_pb2_grpc.TextGenerationServiceServicer에 대한 service 명세는 protobuf 파일에서 확인할 수 있겠죠. 

여기에서는 일단 살펴 보지는 않겠습니다. 

### __init__()

__init__() method는 다음과 같이 구현되어 있습니다.

주어진 model, cache, server URL을 설정하고, model device가 cuda인 경우에는 inference mode를 설정합니다.

```python
    def __init__(
        self,
        model: Model,
        cache: Cache,
        server_urls: List[str],
    ):
        self.cache = cache
        self.model = model
        # Quantize is resolved during model loading
        self.quantize = model.quantize
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)
```

Model 객체는 텍스트 생성을 처리하는 model이겠죠. GPT나 BERT와 같은 모델일 것입니다. 

Cache 객체는 어떤 역할을 하는지는 현재 코드만으로는 알 수 없습니다. 코드를 좀 더 살펴 보면서 알아 보겠습니다. 

server_urls는 server의 URL을 나타냅니다. 마찬가지로 좀 더 코드를 살펴봐야 겠네요. 

마지막으로, model의 device type이 "cuda" 인 경우, PyTorch의 _InferenceMode를 활성화하여, 텍스트 생성 서비스가 실행되는 동안 모델을 추론 모드로 강제 설정합니다

**추론 모드(Inference Mode)**는 모델이 학습 모드와는 다른 방식으로 실행되도록 합니다. GPU 환경에서 모델의 추론 성능을 최적화 할 수 있습니다.

주석을 보면, GLOO(분산 컴퓨팅 프레임워크)를 사용하는 CPU 환경에서는 이 모드가 잘 동작하지 않으므로, CUDA 장치에서만 사용되도록 설정되어 있습니다. 

어떤 이유인지 궁금하네요. 😁



