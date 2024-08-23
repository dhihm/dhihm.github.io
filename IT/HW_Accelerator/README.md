# HW Accelerator

source: `{{ page.path }}`

{% assign recent_posts = site.posts | sort: 'date' | reverse | slice: 0, 3 %}
<ul>
  {% for post in recent_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>