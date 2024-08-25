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

주석을 보면, [GLOO: 분산 컴퓨팅 프레임워크](https://github.com/facebookincubator/gloo)를 사용하는 CPU 환경에서는 이 모드가 잘 동작하지 않으므로, CUDA 장치에서만 사용되도록 설정되어 있습니다. 

어떤 이유인지 궁금하네요. 😁 

PyTorch의 inference mode에 대해서 좀 더 알아 보겠습니다. 

inference mode에서는 Autograd와 연관된 연산 기록 비활성, 불필요한 graph 생성 및 메모리 할당 방지 등을 수행하기 때문에, 메모리 사용량 감소 및 연산 속도 향상을 가져올 수 있습니다. 

주로 training에서 필요한 연산들을 제외하고, 추론에 필요한 연산만 수행하도록 최적화되어 있습니다. 

### Health()

이 method를 사용해서, 서버의 상태를 확인할 수 있습니다. 

비동기 방식으로 구현되어 있네요. 

```python
async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()
```

간단한 방식으로, torch.zeros()를 호출하여, CUDA 장치가 정상적으로 동작하는지 확인합니다.

그 후 generate_pb2.HealthResponse()를 반환하여, 서버의 상태를 확인할 수 있도록 합니다. 

```python
    return generate_pb2.HealthResponse()
```

이 method는 매우 간단하게 구현되어 있고, cuda device의 경우에는 직접적으로 tensor를 생성하여 동작을 확인하기 때문에, 

빈번하게 호출되지 않는다면, 성능에 큰 영향을 미치지 않을 것입니다. 

단순 동작 확인 외 다른 정보를 확인하고 싶을 수 있을 것 같은데, 이 부분은 어떻게 지원하고 있는지 좀 더 살펴 보아야 할 것 같습니다.


### Warmup()

이 method는 모델의 최적화를 위해, 사용 가능한 최대 토큰 수를 사전 계산하여 반환하는 역할을 합니다.

양자화 방식이 "exl2"나 "gptq"인 경우에는, 추가 동작을 먼저 하고 있습니다. 

GPTQ(저비트 양자화)와 같은 특정 양자화 방식은 고유의 연산 커널을 필요로 하며, 이 커널들은 model 로드 후에 최종 형태가 결정됩니다. 

이를 위해 `create_exllama_buffers`를 호출하여 필요한 버퍼를 할당합니다.

create_exllama_buffers() 함수는 GPTQ와 같은 양자화된 모델을 사용할 때 필요한 버퍼를 생성하는 함수입니다. 

이 함수는 특히 ExLlama 커널(ExLlama kernels)과 관련된 메모리 버퍼를 설정하는 역할을 합니다. 

ExLlama는 GPT 모델을 효율적으로 양자화하고, 이를 빠르게 추론할 수 있도록 돕는 최적화된 커널입니다.

버퍼를 할당하기 위한 파라미터로 `max_prefill_tokens`을 사용하고 있습니다. 

사전에 채울 수 있는 최대 토큰 수에 맞춰 메모리 버퍼를 생성하는 것이죠.

버퍼를 할당하기 전에 `set_device`를 호출하여, 모델이 사용하는 device를 설정하고, 해당 device에 맞게 버퍼를 생성합니다.

```python
async def Warmup(self, request, context):
    if self.quantize in {"exl2", "gptq"}:
        try:
            # When using GPTQ, Exllama kernels need some global kernels
            # For which we have the finale shapes only after the model has loaded
            # This will allocate those buffers.
            from text_generation_server.layers.gptq import (
                create_exllama_buffers,
                set_device,
            )

            set_device(self.model.device)
            create_exllama_buffers(request.max_prefill_tokens)
        except ImportError:
            pass
```

다음으로는 model의 batch 초기화와 관련된 중요한 작업을 수행합니다.

요청된 데이터(request.batch)를 model이 처리할 수 있는 형태로 변환한 후, 해당 batch를 사용하여 model의 warmup 작업을 수행하는 과정입니다. 

이 조건문을 통해 현재 모델의 batch_type이 VLM_BATCH_TYPES에 포함되어 있는지를 확인합니다.

VLM_BATCH_TYPES는 특정 모델 배치 유형을 나타내는 집합(set)으로, 이 모델들이 특별한 초기화 과정을 필요로 한다는 것 알 수 있습니다.

```python
    if self.model.batch_type in VLM_BATCH_TYPES:
```

VLM_BATCH_TYPES에 포함되어 있는 경우, model의 batch 초기화를 수행합니다.

```python
    batch = self.model.batch_type.from_pb_processor(
        request.batch,
        self.model.tokenizer,
        self.model.processor,
        self.model.model.config,
        self.model.dtype,
        self.model.device,
    )
```

from_pb_processor라는 메서드를 사용하여 배치를 초기화 하고 있는데, from_pb() 보다 self.model.processor와 self.model.model.config를 추가로 사용하고 있습니다.

이 method는 gRPC request로부터 받은 데이터(protobuf)를 model이 처리할 수 있는 형태로 변환하는 역할을 합니다.

아래 코드를 참고해보면, processor와 config가 사용되는 부분을 확인할 수 있습니다. 

[참고 자료: vlm_causal_lm.py](https://github.com/huggingface/text-generation-inference/blob/f3c5d7d92f005c3cd6a801a33334fb9ba32f55f8/server/text_generation_server/models/vlm_causal_lm.py)

```python
    @classmethod
    def from_pb_processor(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "VlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config
        )
        batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)
        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device)
            if "pixel_attention_mask" in image_inputs:
                batch.pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
                    device=device
                )
            else:
                batch.pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                batch.image_sizes = image_inputs["image_sizes"].to(device=device)
            else:
                batch.image_sizes = None
        else:
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
        return batch
```

batch type이 VLM_BTACH_TYPES에 포함되지 않는 경우에는, 아래와 같이 batch를 초기화합니다.

```python
    else:
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.dtype, self.model.device
        )
```

그 후 초기화된 batch를 사용하여 model의 warmup 과정을 수행합니다. 

이 warmup 과정은 model이 이후에 들어올 데이터를 효율적으로 처리할 수 있도록 필요한 준비 작업을 수행합니다.

warmup 과정에서 계산된 최대 지원 토큰 수가 max_supported_total_tokens 변수에 저장됩니다. 이는 모델이 효율적으로 처리할 수 있는 입력의 크기를 나타냅니다.

```python
    max_supported_total_tokens = self.model.warmup(batch)
```

마지막으로, generate_pb2.WarmupResponse 객체를 생성하여 반환합니다.

gRPC 서버에서 클라이언트에게 응답을 보내는 부분으로, WarmupResponse라는 특정 응답 메시지를 생성하고,

그 안에 이전에 계산된 max_supported_total_tokens 값을 포함시킵니다.

클라이언트는 이 response를 받아 model이 워밍업을 성공적으로 완료했는지, 그리고 얼마 정도의 입력 크기까지 지원할 수 있는지를 확인할 수 있습니다.

```python
    return generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )
```

### Prefill()

이 Prefill method는 model에 입력 데이터를 주입하고 토큰을 생성하는 과정을 처리합니다.

```python
if self.model.batch_type in VLM_BATCH_TYPES:
    batch = self.model.batch_type.from_pb_processor(
        request.batch,
        self.model.tokenizer,
        self.model.processor,
        self.model.model.config,
        self.model.dtype,
        self.model.device,
    )
else:
    batch = self.model.batch_type.from_pb(
        request.batch, self.model.tokenizer, self.model.dtype, self.model.device
    )
```

warmup() method와 유사하게, VLM_BATCH_TYPES에 포함되어 있는 경우에는 from_pb_processor()를 사용하여 batch를 초기화하고 있습니다.

그 이후, generate_token method를 사용하여 초기화된 batch로 model에서 토큰을 생성합니다. 

이 method는 `generations`, `next_batch`, `timings`를 반환하고 있습니다. 

`generations`는 생성된 토큰을 나타내며, `next_batch`는 다음 처리에 사용할 batch를 나타냅니다.

`timings`는 각 단계의 실행 시간을 나타냅니다.

```python
    generations, next_batch, timings = self.model.generate_token(batch)
```

다음 요청에서 사용할 next_batch를 cache에 저장합니다. 이는 model이 상태를 유지하면서 연속적인 요청을 처리할 수 있도록 합니다.

```python
    self.cache.set(next_batch)
```

마지막으로, PrefillResponse를 생성하고 반환합니다. 

generation.to_pb()를 사용하여, 생성된 토큰을 protobuf 형태로 변환하고, next_batch.to_pb()를 사용하여 다음 batch를 protobuf 형태로 변환합니다.

그리고 각 단계의 실행 시간을 timings에 저장하고, 전체 실행 시간을 계산하여 반환합니다.

```python
    return generate_pb2.PrefillResponse(
        generations=[generation.to_pb() for generation in generations],
        batch=next_batch.to_pb() if next_batch else None,
        forward_ns=timings[0],
        decode_ns=timings[1],
        total_ns=time.time_ns() - start,
    )
```

prefill은 text generation 과정에서 model의 입력 데이터를 처리하고, 초기 결과를 생성하는 첫 단계로, 이후 작업의 성능과 품질에 영향을 끼치게 되기 때문에, 

매우 중요한 단계입니다. 텍스트 생성 모델은 일반적으로 sequence data (text)를 처리하는데, 이 데이터는 모델에 주입되기 전에 특정한 형식으로 변환되어야 합니다.

바로 텍스트를 토큰화(tokenization) 하고, 필요한 경우 padding, truncation, special token 추가 등의 전처리를 수행하는 것이죠.

prefill 과정에서는 클라이언트가 요청한 텍스트 데이터를 받아 이를 tokenizer를 통해 토큰으로 변환하고, 필요에 따라 전처리기를 사용하여 데이터를 정규화하거나 모델의 설정에 맞게 조정합니다.

### Decode()

Decode()는 이전에 처리된 배치를 받아서 이를 디코딩하고 새로운 토큰을 생성하는 역할을 합니다. 

이 method는 여러 batch를 결합하여 처리하거나 단일 batch를 처리한 결과를 반환하는 과정을 포함합니다.

맨 처음으로, 클라이언트로부터 전달된 `request.batches`가 비어 있는지 확인합니다. 

batch가 하나도 없는 경우 예외(ValueError)를 발생시켜, 최소한 하나의 batch는 제공되어야 함을 명시적으로 알려줍니다.

```python
    if len(request.batches) == 0:
        raise ValueError("Must provide at least one batch")
```

그리고, `request.batches`에 있는 각 batch의 ID를 기반으로 cache에서 batch를 복원(pop)합니다.

```python
    batches = []
    for batch_pb in request.batches:
        batch = self.cache.pop(batch_pb.id)
        if batch is None:
            raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
        batches.append(batch)
```

각 batch를 batches 리스트에 추가합니다.

다음으로, 처리할 데이터 (복구된 batches)가 있는지 확인합니다. 없으면, 에러를 발생시킵니다.

```python
    if len(batches) == 0:
        raise ValueError("All batches are empty")
```

처리할 데이터가 있는 경우, 즉 복원된 batch가 하나 이상이면, 이 batch들을 하나로 결합합니다. (concatenate)

batch가 하나뿐이면 결합 과정이 필요 없으므로, 첫 번째 batch를 그대로 사용합니다.

```python
    if len(batches) > 1:
        start_concat = time.time_ns()
        batch = self.model.batch_type.concatenate(batches)
        concat_ns = time.time_ns() - start_concat
    else:
        batch = batches[0]
        concat_ns = None
```

그리고 결합된(또는 단일) batch를 사용하여 model에서 토큰을 생성합니다. generate_token 메서드의 반환 값은 위에서 설명했습니다. 

```python
    generations, next_batch, timings = self.model.generate_token(batch)
```

이후, 다음 요청에서 사용할 next_batch를 cache에 저장합니다.

```python
    self.cache.set(next_batch)
```

마지막으로, DecodeResponse를 생성하고 반환합니다.

```python
    return generate_pb2.DecodeResponse(
        generations=[generation.to_pb() for generation in generations],
        batch=next_batch.to_pb() if next_batch else None,
        concat_ns=concat_ns,
        forward_ns=timings[0],
        decode_ns=timings[1],
        total_ns=time.time_ns() - start,
    )
```

### serve()

이 코드에서는 내부에서 비동기로 실행되는 serve_inner 함수를 통해 텍스트 생성 model을 기반으로 하는

gRPC 서버를 설정하고 실행하는 전체적인 흐름을 보여줍니다. 이 서버는 우리가 예상하는 것처럼 model을 초기화하고, 클라이언트의 요청을 처리하는 역할을 합니다.

```python
    def serve(
        model_id: str,
        lora_adapters: Optional[List[AdapterInfo]],
        revision: Optional[str],
        sharded: bool,
        quantize: Optional[str],
        speculate: Optional[int],
        dtype: Optional[str],
        trust_remote_code: bool,
        uds_path: Path,
        max_input_tokens: int,
    ):
```

먼저 serve() method의 paraemter를 살펴보겠습니다.

model_id: 모델의 고유 식별자. 로드할 모델을 지정하는 데 사용됩니다. 

lora_adapters: LoRA(LoRA는 모델 파인튜닝을 위한 어댑터) 어댑터의 정보 목록. 특정 모델의 성능을 조정하는 데 사용됩니다. 

sharded: 모델이 여러 노드에 분산되어 있는지 여부를 나타내는 부울 값. 분산 학습을 위한 설정입니다. 

quantize: 모델을 양자화할 때 사용하는 방법. 성능 최적화 또는 메모리 사용량 절감을 위해 모델을 양자화할 수 있습니다.

speculate: 사전 추측 작업을 위한 선택적 정수 값. 이 값은 미래의 계산을 미리 수행하여 성능을 최적화하는 데 사용될 수 있습니다.

dtype: 모델이 사용할 데이터 타입(예: float32, float16 등).

trust_remote_code: 외부에서 다운로드한 모델 코드가 신뢰할 수 있는지를 결정하는 값. 보안을 위해 사용됩니다.

uds_path: Unix 도메인 소켓 경로. 서버가 사용하는 소켓의 경로를 지정합니다.

max_input_tokens: 입력으로 받을 수 있는 최대 토큰 수 입니다. 

serve_inner() 에서는 위에서 설명한 parameter를 사용하여 gRPC 서버를 설정하고 실행합니다.

먼저 서버의 설정 과정에서 Unix 도메인 소켓을 사용하는 gRPC 서버의 URL을 생성하고 설정하는 역할을 합니다. 

특히, 서버가 분산 처리(샤딩) 환경에서 작동하는 경우, 각 노드에 고유한 URL을 할당하는 과정을 처리합니다.

```python
    unix_socket_template = "unix://{}-{}"
        adapter_to_index = {}
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

```

환경 변수 WORLD_SIZE는 전체 노드의 수를 나타내고, RANK는 현재 노드의 순서(index)를 나타냅니다.

server_urls 리스트는 각 노드에 대해 고유한 서버 URL을 생성합니다.

각 노드의 인덱스(rank)에 따라 uds_path와 rank를 결합한 Unix 도메인 소켓 URL이 생성됩니다. 

예를 들어, uds_path가 /tmp/socket이고 rank가 1이라면, URL은 unix:///tmp/socket-1이 됩니다.

각 노드는 자신만의 로컬 URL을 가져야 하므로, server_urls 리스트에서 자신의 인덱스(rank)에 해당하는 URL을 선택합니다.

sharded가 False이면, 서버는 단일 인스턴스에서 동작하니까, 노드 인덱스(rank)는 0으로 설정되며, 단일 서버 URL만 생성됩니다.

server_urls 리스트에는 단 하나의 URL만 포함됩니다.

```python
    else:
        local_url = unix_socket_template.format(uds_path, 0)
        server_urls = [local_url]
```

그 이후, get_model_with_lora_adapters()를 호출하여 model을 초기화하고, 이 과정에서 발생할 수 있는 예외를 처리합니다.

```python
    model = get_model_with_lora_adapters(
        model_id,
        lora_adapters,
        revision,
        sharded,
        quantize,
        speculate,
        dtype,
        trust_remote_code,
        max_input_tokens,
        adapter_to_index,
    )
```

model이 초기화 되면, gRPC server를 초기화 합니다. 

먼저 set_adapter_to_index()를 호출하여, 어댑터와 인덱스를 매핑하는 딕셔너리를 설정합니다.

이 딕셔너리는 어댑터와 인덱스를 매핑하는 정보가 저장되어 있으며, model의 일부 설정에 사용됩니다.

LoRA 어댑터와 같은 model 어댑터는 특정 인덱스에 할당되어 model의 특정 부분에 적용됩니다. 

set_adapter_to_index는 이러한 인덱스 설정을 model에 적용하는 작업을 수행합니다. 

```python
    set_adapter_to_index(adapter_to_index)
        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ],
            options=[
                # Set the maximum possible message length: i32::MAX
                ("grpc.max_receive_message_length", (1 << 31) - 1)
            ],
        )
```

model이 여러 개의 어댑터를 사용할 때, 각 어댑터가 model의 어떤 부분에 연결되어야 하는지를 지정해야 합니다.

이 set_adapter_to_index()는 그 설정을 관리하며, 이후 모델이 올바르게 작동할 수 있도록 준비하는 것입니다. 

다음으로, gRPC 서버를 생성합니다. aio.server는 gRPC 서버를 비동기적으로 생성하는 함수입니다.

server를 생성 할 때 interceptors를 설정하고 있습니다. 

interceptor는 gRPC 호출 전에 실행되는 코드를 추가하여 요청/응답을 수정하거나, 로깅, 모니터링, 예외 처리 등의 추가 작업을 수행할 수 있도록 합니다.

이름으로 추측해보면, exception이 발생 했을 때 사용하는 ExceptionInterceptor와 OpenTelemetry를 사용하여 서버에서 발생하는 트랜잭션을 추적하고

모니터링 하는 용도로 사용 될 것 같은 UDSOpenTelemetryAioServerInterceptor를 설정하고 있습니다.

option으로는 gRPC가 처리할 수 있는 최대 메시지 크기를 2GB로 설정하고 있네요.

그리고 generate_pb2_grpc를 사용하여 `TextGenerationService`라는 gRPC 서비스 구현을 서버에 등록합니다. 

```python
    generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
        TextGenerationService(model, Cache(), server_urls), server
    )
```

그 후, 서비스 reflection을 활성화하고, 서버를 특정 주소(포트)에서 실행할 수 있도록 설정하는 작업을 수행합니다.

reflection은 클라이언트가 서버의 서비스와 메서드에 대한 메타데이터를 동적으로 조회할 수 있게 해주며, 서버는 지정된 로컬 URL에서 클라이언트 요청을 받아들일 있게 됩니다. 

이를 통해 클라이언트가 서버의 API를 미리 알지 못하더라도, 서버에서 제공하는 기능을 자동으로 탐색하고 사용할 수 있게 됩니다.

```python
    SERVICE_NAMES = (
        generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(local_url)
```

`SERVICE_NAMES`는 서버에서 reflectoin을 활성화할 서비스의 이름들을 담고 있는 튜플입니다.

`generate_pb2` 모듈의 `DESCRIPTOR` 객체를 사용하여 TextGenerationService 서비스의 전체 이름(full name)을 가져오는데,

DESCRIPTOR는 gRPC 서비스와 메시지에 대한 메타데이터를 포함하는 객체로, .proto 파일에서 정의된 서비스와 메시지 정보를 담고 있습니다.

`add_insecure_port()`를 사용하여, TLS/SSL을 사용하지 않고 서버를 실행할 포트를 설정합니다.

실제 프로덕션이 아니기 때문에, 보안을 고려하지 않고 간단하게 실행할 수 있도록, 인증서를 사용하지 않는 인증 없는(insecure) 포트를 사용하고 있는 거겠죠.

다음은 서버를 비동기적으로 시작하고, 서버가 실행되는 동안 지속적으로 상태를 모니터링하여 서버가 중단될 때까지 실행을 유지하는 코드 입니다. 

```python
    await server.start()

    logger.info("Server started at {}".format(local_url))
    signal_handler = SignalHandler()
    while signal_handler.KEEP_PROCESSING:
        await asyncio.sleep(0.5)
```

이 함수는 비동기 함수이기 때문에 await 키워드를 사용하여 호출되며, 이로 인해 이벤트 루프가 차단되지 않고 다른 비동기 작업을 계속 수행할 수 있습니다. 

그리고 `SignalHandler` 객체를 생성하여 서버가 종료될 때의 신호를 처리합니다.

`SignalHandler` 내부에 `KEEP_PROCESSING`이라는 플래그가 포함되어 있어, 이 플래그가 False로 변경되면 서버가 종료됩니다

이제 다시 `serve()` 함수로 돌아와서, 마지막 구현을 확인하겠습니다. 

```python
     asyncio.run(
        serve_inner(
            model_id,
            lora_adapters,
            revision,
            sharded,
            quantize,
            speculate,
            dtype,
            trust_remote_code,
        )
    )
```

 `asyncio.run()`을 사용하여 `serve_inner` 비동기 함수를 실행하는 역할을 합니다. 
 
 `asyncio.run()`은 Python의 비동기 프로그래밍을 처리하는 이벤트 루프를 시작하고, 주어진 비동기 코루틴을 실행하여 최종 결과를 반환합니다. 

 다음 리뷰에서는 이 서버와 함께 동작하는 클라이언트를 리뷰해 보겠습니다. 😋
