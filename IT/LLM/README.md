# LLM

source: `{{ page.path }}`

LLM Inference Engine 설계 및 구현에 관련한 내용을 정리합니다. 

{% assign recent_posts = site.posts | sort: 'date' | reverse | slice: 0, 3 %}
<ul>
  {% for post in recent_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>