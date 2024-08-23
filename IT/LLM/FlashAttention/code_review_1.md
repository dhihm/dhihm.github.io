# Flash Attention Code Review 1

### flash_attention_interface.py

get_block_size() in `flash_attention_interface.py` 
(https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)

```python
def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64
```

이 함수는 다음과 같은 parameter를 사용하고 있습니다. 
- `head_dim`는 attention head의 차원입니다.
- `is_dropout`은 dropout이 사용되는지 여부를 나타내는 플래그입니다.
- `is_causal`은 attention이 인과적인지 여부를 나타내는 플래그입니다.
- 이 함수는 입력 매개변수를 기반으로 CUDA 커널을 위한 블록 크기를 반환합니다.
- 블록 크기는 `head_dim`과 GPU 아키텍처에 따라 결정됩니다.

github 에서 심볼 검색해 보면, 이 함수는 실제 사용되고 있지는 않는 것 같습니다. 하지만 이 함수에 대해서 정리하고자 한 이유는 다음과 같습니다. 

먼저 이 함수에서 말하고 있는 `block` 이라는 것은 CUDA 프로그래밍에서 사용되는 개념입니다. CUDA 프로그래밍에서 `block`은 GPU에서 동시에 실행되는 thread group입니다.

이 함수는 입력 parameter를 기반으로 CUDA 커널을 위한 `block` 크기를 반환하는데, `block` 크기는 `head_dim`과 GPU 아키텍처에 따라 결정됩니다.

### CUDA Thread, Block, Grid

CUDA 프로그래밍에서는 thread, block, grid라는 개념이 사용됩니다.

- Thread: CUDA 프로그램의 가장 기본적인 실행 단위입니다. 각 thread는 고유한 ID를 가지며, 이 ID를 통해 thread는 다른 thread와 구분됩니다.
- Block: thread의 그룹입니다. block은 1차원, 2차원, 3차원으로 구성될 수 있습니다. 각 block은 고유한 ID를 가지며, 이 ID를 통해 block은 다른 block과 구분됩니다.
- Grid: block의 그룹입니다. grid는 1차원, 2차원, 3차원으로 구성될 수 있습니다.


### GPU 아키텍처

- Warp: GPU에서 동시에 실행되는 thread의 그룹입니다. 각 warp는 32개의 thread로 구성되어 있습니다.
- 레지스터 파일 크기: 각 thread는 레지스터 파일을 사용합니다. 레지스터 파일 크기는 GPU 아키텍처에 따라 다릅니다.
- Shared Memory: block 내의 thread들이 공유하는 메모리입니다. Shared Memory는 block 내의 thread들이 공유하는 메모리입니다.

[Warp, Block, Grid, Thread, Shared Memory, Register](/Users/dhihm/source/dhihm.github.io/IT/LLM/CUDA/cuda_review_1.md)

특정 block size를 사용하면, register나 shared memory가 고갈되는 문제가 발생할 수 있습니다. 이 함수는 이러한 문제를 방지하기 위해 block size를 결정하는데 사용됩니다.
큰 block size는 더 많은 thread를 동시에 실행할 수 있지만, register나 shared memory가 고갈되는 문제가 발생할 수 있습니다.
작은 block size는 register나 shared memory가 고갈되는 문제를 방지할 수 있지만, 더 많은 block을 사용해야 하기 때문에 성능이 저하될 수 있습니다.

그렇다면, head_dim과 GPU 아키텍처에 따라 block size를 결정하는 이유는 무엇일까요?

3가지로 그 이유를 정리할 수 있습니다.

1. 리소스 효율화
2. GPU Architecture에 따른 최적화
3. 병렬 처리 극대화

#### 리소스 효율화

- Register와 Shared Memory 제약: GPU에서 각 thread는 register와 shared memory를 사용하는데, head_dim이 클 수록 각 thread가 처리해야할 데이터가 많아지기 때문에, 이에 따라 필요한 resource (register, shared memory)도 증가합니다. 큰 block size를 사용하면, 한 번에 실행되는 thread 수가 많아지지만, 이를 지원하기 위한 resource도 증가합니다. 
- Thread 병렬 처리: 작은 block size를 사용하면, 실행되는 thread 수가 줄어들어 register와 shared memory 사용량을 줄일 수 있지만, 이는 병렬 처리 효율성을 낮추는 요인이 될 수 있습니다. 반면 큰 block size를 사용하면, 더 많은 thread를 동시에 실행할 수 있지만 register와 shared memory 사용량이 증가하여 resource가 고갈됨으로써 성능이 저하될 수 있습니다.

#### GPU Architecture에 따른 최적화

- SM 구조와 최적화: GPU Architecture에 따라 Streaming Multiprocessor (SM)의 구조가 달라, register, shared memory, warp의 수 scheduling 방식 등에서 차이가 있을 . 수있습니다. 예를 들어 Ampere architecture에서는 register 파일 크기가 256KB이고, shared memory가 164KB입니다. Volta architecture에서는 register 파일 크기가 256KB이고, shared memory가 96KB입니다. 이러한 architecture에 따른 차이를 고려하여 block size를 결정할 수 있습니다.
- Architecture별 최적 성능: 각 GPU architecture는 최적의 block size가 존재합니다. 예를 들어 Ampere architecture에서는 head_dim이 큰 경우에도 더 큰 block size를 사용할 수 있지만, Volta architecture에서는 head_dim이 큰 경우에는 작은 block size를 사용하는 것이 성능이 더 좋을 수 있습니다. 이는 단순히 register와 shared memory 크기 뿐 아니라, warp scheduling, thread block scheduling 등과 같은 architecture 최적화에 기인합니다. 

#### 병렬 처리 극대화

- Thread divergence 방지: head_dim이 클 수록, 연산 중 thread 간 divergence가 발생할 가능성이 커지기 때문에, 이를 방지할 수 있는 적절한 block size를 선택하여, 병렬 처리 효율성을 극대화할 수 있습니다.
  - Thread divergence: warp 내의 thread가 서로 다른 branch를 타게 되어, 서로 다른 instruction을 실행할 때 발생하는 성능 저하 문제를 의미합니다.
    - warp 내의 모든 thread는 동일한 instruction을 실행해야 하므로, 조건문 등에서 다른 분기를 선택한 thread들은 모두 대기해야 합니다. thread divergence가 발생하면, GPU는 각 분기별로 warp를 직렬화하여 실행하게 되어, 병렬 실행의 장점이 감소되는 문제가 발생합니다. 

 
```c++
__global__ void exampleKernel(int *data, int threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (data[idx] > threshold) {
        // 이 코드 경로는 일부 스레드만 실행
        data[idx] = data[idx] * 2;
    } else {
        // 이 코드 경로는 다른 스레드들이 실행
        data[idx] = data[idx] + 1;
    }
}
```

- Memory coherency: block size가 적절하지 않으면, memory coherency 문제가 발생할 수 있습니다. Memory coherency 문제는 memory access가 동시에 발생할 때, memory access가 출동하여 데이터가 손실되는 문제입니다. 이를 방지하기 위해 적절한 block size를 선택하여야 합니다. block size가 클 때 memory coherency 문제가 발생하는 이유는, 주로 global memory access pattern의 비효율성 때문입니다. 이를 좀 더 자세히 알아보겠습니다. 

GPU memory hierarchy는 다음과 같이 구성되어 있습니다.

- Global memory: 모든 thread가 접근할 수 있는 가장 큰 memory입니다. global memory는 latency가 높고, bandwidth가 낮습니다.
- Cache memory: global memory에 접근하는 latency를 줄이기 위해 사용되는 memory입니다. cache memory는 latency가 낮고, bandwidth가 높습니다. (L1, L2)

GPU에서 warp 단위로 memory access가 발생할 때 warp 내 모든 thread가 동일한 memory access pattern (coalesced memory access)을 가질 때 memory coherency가 잘 유지되며, memory access 성능이 최적화됩니다. block size가 큰 경우에는 많은 thread를 포함하므로, 각 thread가 다루는 데이터가 분산될 가능성이 높아, warp 내 thread 간 서로 다른 memory access pattern을 가질 가능성이 높아집니다. 이러한 경우 여러 번의 memory access가 요청되어 memory access 효율이 떨어지고, 대역폭이 낭비되는 것입니다. 

head_dim과 이러한 GPU 구조는 어떤 관계가 있을까요?
head_dim은 attention head의 차원을 나타내는데, head_dim이 클수록 각 thread가 처리해야할 데이터가 많아지고, 필요한 resource (register, shared memory)도 증가합니다.

따라서 이 함수에서는 이러한 리소스 효율화, GPU Architecture에 따른 최적화, 병렬 처리 극대화를 고려하여 block size를 결정하고 있는데, 실제 사용되고 있지는 않는 것 같아서 flash attention 코드에서 어떻게 사용되는지 좀 더 확인해 봐야겠습니다. :stuck_out_tongue:

source: `{{ page.path }}`
