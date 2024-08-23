# Flash Attention Code Review 1

## Code Review

### General

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

먼저 이 함수는 다음과 같은 매겨변수를 사용하고 있습니다. 
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

특정 block size를 사용하면, register나 shared memory가 고갈되는 문제가 발생할 수 있습니다. 이 함수는 이러한 문제를 방지하기 위해 block size를 결정하는데 사용됩니다.
큰 block size는 더 많은 thread를 동시에 실행할 수 있지만, register나 shared memory가 고갈되는 문제가 발생할 수 있습니다.
작은 block size는 register나 shared memory가 고갈되는 문제를 방지할 수 있지만, 더 많은 block을 사용해야 하기 때문에 성능이 저하될 수 있습니다.

그렇다면, head_dim과 GPU 아키텍처에 따라 block size를 결정하는 이유는 무엇일까요?

TBD


source: `{{ page.path }}`
