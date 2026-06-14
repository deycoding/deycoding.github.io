---
layout: default
title: Home
---

# deycoding

Building LLMs from scratch. Thoughts on AI, architecture, and engineering.

## Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) - *{{ post.date | date: "%B %d, %Y" }}*
{% endfor %}
