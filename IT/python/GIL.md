---
layout: post
title: "Python async programming"
date: 2024-10-09 08:44:00 +0900
categories: IT
author: dh.ihm
---

# Python의 GIL (Global Interpreter Lock)

이 페이지에서는 python의 GIL에 대해서 알아보고자 합니다. 

GIL은 Python 인터프리터가 한 번에 하나의 스레드만 Python 바이트코드를 실행할 수 있도록 보장하는 뮤텍스(mutex)입니다

GIL이라는 것이 대체 왜 필요한가 부터 시작해 보겠습니다. 

먼저 CPython이라는 것에 대해서 이해가 조금 필요합니다. 간단하게 설명하면, CPython이라는 것은 python 코드가 실행될 수 있도록 하는

컴파일러와 VM(Virtual Machine), 그리고 관련된 모든 라이브러리들입니다. 

이를 좀 더 고급지게 표현하면 CPython은 Python programming language의 구현체이다. 라고 할 수 있습니다. 

이 CPython이라는 것이 GIL을 이용해서 multi thread 환경에서도 python 객체에 한 번에 하나의 thread만 접근하도록

Lock을 걸고 있는 것이죠. 

CPython의 구현 방식 및 철학과 관련이 있는 것 같습니다. 이름에서 알 수 있듯이 CPython은 C언어로 구현되어 있고, 

reference counting 방식을 사용해서 메모리 관리를 하고 있습니다.

multi thread 환경에서 이 reference count 관련한 문제점들을 안전하게 보장하는 방법이 쉽지는 않았겠죠. 

마찬가지로 많은 C언어로 작성된 확장이나 라이브러리들에서도 thread safety를 보장해야 하기 때문에 구현의 난이도가 상승하게 되고요. 

그래서 CPython은 아예 문제를 입구컷 하겠다라는 철학으로, 한 번에 하나의 thread만 python object에 접근하도록 하여, 

내부 상태를 보호하기 위한 복잡한 메커니즘들을 생략할 수 있는 방법, 즉 Global interpreter Lock을 사용한 것입니다. 

다만 이렇게 하기 때문에 multi thread의 이점을 제대로 활용할 수 없는 문제가 생기고, 

Faster CPython 같은 프로젝트를 통해서 성능 개선을 위한 노력을 하고 있다고 합니다. 

Rust를 이용해 보면 더 좋을 것 같다라는 생각이 들긴 하네요. 😁


그럼 python에서는 병렬처리를 어떻게 해야 할까요???

여러 라이브러리 등이 이를 지원하고 있는데, 결국 핵심은 multi thread 대신에 multi process를 사용해서 GIL의 제약을 피하는 것입니다. 

다만 멀티 프로세스를 사용하게 되면서 메모리 사용량이 좀 더 늘어나게 되고, 프로세스간 데이터 공유가 필요한 경우 IPC (Inter-Process Communication) 등을

사용해야 하기도 하고, 디버깅도 더 어려워 질 수 있는 문제들이 있을 수 있습니다. 😂
