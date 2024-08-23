# Tensor Parallelism for faster inference on multiple GPUs

Tensor Parallelism에 대해서 먼저 살펴 보겠습니다. 

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

[참고자료 - Tensor Parallelism](https://github.com/huggingface/text-generation-inference/blob/main/docs/source/conceptual/tensor_parallelism.md)

실제 코드에서는 어떻게 구현되어 있는지 살펴보겠습니다.


source: `{{ page.path }}`