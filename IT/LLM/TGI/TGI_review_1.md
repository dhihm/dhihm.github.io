# Tensor Parallelism for faster inference on multiple GPUs

Tensor Parallelism에 대해서 먼저 살펴 보겠습니다. 

[참고자료 - Tensor Parallelism](https://github.com/huggingface/text-generation-inference/blob/main/docs/source/conceptual/tensor_parallelism.md)

## Tensor Parallelism

Tensor Parallelism은 모델을 여러개의 GPU에 분산하여 학습하거나 추론하는 방법입니다.

이 방법은 모델의 크기가 커지면서 메모리가 부족해지는 문제를 해결하기 위해 사용됩니다.

Tensor Parallelism의 기본 아이디어는 다음과 같습니다. 

tesnor와 tensor의 matrix multiplication은 곱해지는 tensor를 분할하여, 각각의 부분을 곱한 다음 결과를 합치는 것과 같기 때문에,
 
tensor를 분할하여 여러 GPU에 분산하여 계산하면, 계산 속도를 높일 수 있다는 것이죠. 

예를 들어, 두 개의 행렬 A와 B가 있다고 할 때:

\[
A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}, \quad B = \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}
\]

A의 행과 B의 열을 각각 곱한 후 더해서 새로운 행렬 C를 만들어야 합니다. 

\[
C = A \times B = \begin{pmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{pmatrix}
\]

여기서:

\[
c_{11} = a_{11} \times b_{11} + a_{12} \times b_{21}, \quad c_{12} = a_{11} \times b_{12} + a_{12} \times b_{22}
\]
\[
c_{21} = a_{21} \times b_{11} + a_{22} \times b_{21}, \quad c_{22} = a_{21} \times b_{12} + a_{22} \times b_{22}
\]


이 계산을 하나의 GPU가 혼자 한다고 생각하면, time(C_{11}) + time(C_{12}) + time(C_{21}) + time(C_{22}) 시간이 걸릴 것입니다.

하지만 이 계산을 4개의 GPU가 나눠서 한다면 time(C_{11}), time(C_{12}), time(C_{21}), time(C_{22}) 중, 

가장 오래 걸리는 계산을 시간 time{C_{result}}이라고 하면, 이 시간과 각각의 계산을 모두 더하는 계산의 시간 time{C_{calc}}이 걸릴 것입니다. 

이 시간이 하나의 GPU에서 계산하는 것보다 훨신 짧을 거라는 것이죠. 

- GPU 1: \( c_{11} = a_{11} \times b_{11} + a_{12} \times b_{21} \)
- GPU 2: \( c_{12} = a_{11} \times b_{12} + a_{12} \times b_{22} \)
- GPU 3: \( c_{21} = a_{21} \times b_{11} + a_{22} \times b_{21} \)
- GPU 4: \( c_{22} = a_{21} \times b_{12} + a_{22} \times b_{22} \)

여기에서 좀 더 생각해볼 문제가 있습니다. 각 GPU가 계산한 결과를 더하기 위해서는, GPU가 서로 통신을 해야 한다는 것이죠. 

이 때 데이터가 이동하는데 시간이 걸리고, 이걸 일반적으로 communication overhead라고 부릅니다.

따라서 각 GPU에 분산하여 수행할 작업이 매우 큰 상태라서 계산 시간이 communication overhead보다 훨씬 크다면, tensor parallelism을 사용하는 것이 효과적일 것입니다.

TGI 소개 페이지에서는 "for faster inference" 라고 설명하고 있는데, 잘 이해가 되지는 않습니다.

하나의 GPU에 로드 될 수 있는 모델을 여러 GPU에 tensor parallelism을 이용해 분산 로드 하면 communication overhead가 발생하게 되어 오히려 느려질 것 같은데 말입니다.

대규모 모델의 경우에는 애초에 하나의 GPU에 로드될 수 없기 때문에, inference 성능을 높이기 위해서가 아니라, 

모델 로드를 하기 위해서 tensor parallelism을 사용하는 것이라고 생각 됩니다.   

실제 코드에서는 어떻게 구현되어 있는지 살펴보겠습니다.

[참고자료 - tensor_paralle.py](https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/tensor_parallel.py)

## Tensor Parallelism 구현 리뷰

총 6개의 class로 구성되어 있습니다.

- LayerConcat class
- SuperLayer class
- TensorParallelHead class
- TensorParallelColumnLinear class
- TensorParallelRowLinear class
- TensorParallelEmbedding class

각 class는 다음과 같은 역할을 합니다.

### LayerConcat class

LayerConcat class는 이름에서 알 수 있듯이, 여러 layer의 출력을 연결(concatenate)하여 하나의 출력으로 반환합니다. 

forward() method에서 입력 받은 tensor x를 각각의 layer에 넣어서 출력을 받은 후, torch.cat()을 이용하여 연결하고, 그 결과를 반환하고 있습니다. 

```python
    def forward(self, x: torch.Tensor):
        outputs = [layer(x) for layer in self.layers]
        return torch.cat(outputs, self.dim)
```

이 class에서 중요한 변수는 `dim`입니다. 

```python
    def __init__(self, layers: Iterable[torch.nn.Module], dim: int = -1):
            """
            `dim` is the dimension along which layer outputs are concatenated.
            """
            super().__init__()
            self.layers = layers
            self.dim = dim
```

`dim`은 torch.cat()을 이용하여 tensor를 연결할 때, 어느 차원을 기준으로 연결할 것인지를 나타냅니다.

이 부분을 이해하기 위해서, tensor의 차원에 대해 살펴 보겠습니다. 

tensor는 다차원 배열로, 각 차원은 서로 다른 `축`을 나타냅니다. 예를 들어 2차원 tensor는 `행`과 `열`로 구성된 배열,

3차원 tensor는 `행`, `열`, `깊이`를 가진 배열이라고 생각할 수 있죠.

tensor의 각 차원에는 index가 있으며, 0부터 시작합니다. 

[batch_size, num_features] 형태의 2차원 tensor를 예로 들면,
- dim=0: batch_size (행)
- dim=1: num_features (열) 을 나타내는 것입니다. 

그리고 torch.cat() 함수는, 여러 tensor를 지정된 차원에서 연결하여 하나의 tensor로 만들어주는 함수입니다.

예를 들면, 위의 [batch_size, num_feature] 형태를 갖고, [2, 3] 크기의 tensor A, B가 있을 때,

\[ A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix}, \quad
   B = \begin{pmatrix} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \end{pmatrix}
\]

dim=0에서 연결하면 A와 B가 행 방향으로 붙게 되어 [4, 3] 크기의 tensor가 됩니다.

\[
\text{concat}(A, B, \text{dim}=0) = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23}
\end{pmatrix}
\]

dim=1에서 연결하면 열 방향으로 붙어서 [2, 6] 크기의 tensor가 됩니다.

\[
\text{concat}(A, B, \text{dim}=1) = \begin{pmatrix}
a_{11} & a_{12} & a_{13} & b_{11} & b_{12} & b_{13} \\
a_{21} & a_{22} & a_{23} & b_{21} & b_{22} & b_{23}
\end{pmatrix}
\]

이것의 의미는, dim=0에서 연결하면 결과 tensor들이 쌓이면서 batch size가 증가하고, dim=1에서 연결하면 같은 batch 내에서 각 feature들이 더해지면서 feature 크기가 커진다는 것이죠. 

따라서 개발자는 이 class를 통해, 각 tensor가 어떻게 연결될지를 세밀하게 제어할 수 있고, 다양한 tensor 연산을 수행할 수 있게 됩니다.

### SuperLayer class

이 class는 단순하게 하나의 linear layer를 포함하는 클래스로, forward() method에서 입력 tensor를 linear layer에 넣어서 출력을 반환합니다.

```python
    def forward(self, x):
        return self.linear.forward(x)
```

이름에서 알 수 있듯이 상속을 통해 확장하기 위한 class 입니다. 

### TensorParallelHead class

TensorParallelHead class는 tensor parallelism에서 모델의 마지막 layer(head)를 병렬로 처리하기 위한 class입니다.

load() method는 모델을 로드할 때 사용할 적절한 파라미터를 설정하고, weight를 분할하여 여러 GPU에 로드합니다.

```python
    def load(config, prefix: str, weights):
        if config.quantize == "exl2":
            try:
                # If the piece and LM head embeddings are shared, we have
                # non-quantized weights...
                weight = weights.get_tensor(f"{prefix}.weight")
            except Exception:
                # ...otherwise they are quantized.
                weight = weights.get_weights_col(prefix)
            should_gather = weights.process_group.size() > 1
        elif weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
                should_gather = True
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        return TensorParallelHead(
            get_linear(weight, bias=None),
            process_group=weights.process_group,
            should_gather=should_gather,
        )
```

이 method에서는 총 3개의 parameter를 받고 있습니다.
- config: 모델 구성을 담고 있는 객체로, 양자화 방식, process group (병렬 처리에 사용하는 GPU 그룹) 등의 설정이 포함되어 있습니다. 
- prefix: weight를 가져올 때 사용할 prefix입니다.
- weights: weight들을 관리하는 객체로, weight를 가져오거나, 분할된 가중치를 결합하는 기능을 제공하고 있습니다. 

조건문에 의한 분기는 총 3가지 경우로 나뉩니다.
1. exl2 양자화 방식을 사용하는 경우
2. process group의 크기가 1보다 큰 경우
3. process group의 크기가 1인 경우

exl2 양자화 방식을 사용하는 경우에는, weight를 가져오는 방식이 다릅니다.

양자화된 경우, weight.get_weights_col() method를 사용하여 weight를 가져오고, 양자화되지 않은 경우, weight.get_tensor() method를 사용하여 weight를 가져옵니다.

```python
    if config.quantize == "exl2":
        try:
            # If the piece and LM head embeddings are shared, we have
            # non-quantized weights...
            weight = weights.get_tensor(f"{prefix}.weight")
        except Exception:
            # ...otherwise they are quantized.
            weight = weights.get_weights_col(prefix)
        should_gather = weights.process_group.size() > 1
```

process group의 크기가 1보다 큰 경우는, weight가 여러 GPU에 분산되어 있을 수 있습니다. (sharding)

이 경우, weight.get_sharded() method를 사용하여 weight를 가져오고, should_gather를 True로 설정합니다.

만약 sharding된 weight를 가져오는 중에 문제가 발생하면 (AssertionError), 전체 weight를 불러 오도록 처리하고 있습니다. 

```python
    elif weights.process_group.size() > 1:
        try:
            weight = weights.get_sharded(f"{prefix}.weight", dim=0)
            should_gather = True
        except AssertionError:
            # If the vocab size is not divisible by number of shards
            # just load the entire thing.
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False
```

process group의 크기가 1인 경우는, weight를 가져오는 방법이 가장 간단합니다. 병렬 처리가 필요 없기 때문이죠. 

```python
    else:
        weight = weights.get_tensor(f"{prefix}.weight")
        should_gather = False
```

마지막으로 get_linear(weight, bias=None)을 통해 weight를 가지고 Linear layer를 생성하고, TensorParallelHead class를 생성하여 반환합니다.

`should_gather`는 나중에 병렬 처리 중 weight를 결합할 필요가 있는지를 나타냅니다. 

```python
    return TensorParallelHead(
            get_linear(weight, bias=None),
            process_group=weights.process_group,
            should_gather=should_gather,
        )
```

forward() method는 input tensor를 병렬 처리하고, 필요 시 여러 GPU의 출력을 모아서 반환합니다.

```python
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input)

        world_size = self.process_group.size()
        if len(input.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]

            if input.shape[0] == 1:
                world_out = input.new_empty(1, out_dim * world_size)
                local_out = input.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input.new_empty(out_dim * world_size, input.shape[0])
                gather_input = input.new_empty(out_dim, input.shape[0])
                local_out = gather_input.T

            torch.mm(input, self.linear.weight.T, out=local_out)
            if SYSTEM == "ipex":
                ipex.distributed.all_gather_into_tensor(
                    world_out, gather_input, group=self.process_group
                )
            else:
                torch.distributed.all_gather_into_tensor(
                    world_out, gather_input, group=self.process_group
                )

            if input.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input)
        world_output = [
            torch.empty_like(output) for _ in range(self.process_group.size())
        ]
        if SYSTEM == "ipex":
            ipex.distributed.all_gather(world_output, output, group=self.process_group)
        else:
            torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output
```

TBD

### TensorParallelColumnLinear class
### TensorParallelRowLinear class
### TensorParallelEmbedding class

source: `{{ page.path }}`