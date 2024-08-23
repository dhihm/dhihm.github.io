---
layout: default
title: Thinking, Writing, and.
---

# Thinking, Writing, and.

**"Thinking, Writing, and."에 오신 것을 환영합니다!**  
이 블로그는 다양한 생각을 공유하고, 글을 쓰며, 비슷한 관심사를 가진 사람들과 교류하기 위해 만들어졌습니다.
Software, 특히 LLM 관련된 내용에 대해 정리한 내용을 공유하고, 가끔은 살아가는 이야기도 함께 나누고자 합니다.

---

## 최근 게시글

{% for post in site.posts limit:5 %}
- [{{ post.title }}]({{ post.url }}) <small>({{ post.date | date: "%Y년 %m월 %d일" }})</small>
{% endfor %}

---

## 카테고리

{% for category in site.categories %}
- [{{ category[0] }}]({{ site.baseurl }}/categories#{{ category[0] }})
{% endfor %}

---

"Thinking, Writing, and."에 방문해 주셔서 감사합니다. 