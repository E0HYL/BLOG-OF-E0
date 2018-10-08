---
layout: post
title: Jekyll + Github Pages = Blog
description: "Let me introduce how the blog works for the first post."
modified: 2018-10-7
tags: [Skills, Blog]
image:
  feature: abstract-1.jpg
  credit: DarGadgetZ
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

> Though written in English, links here that I attach are almost Chinese (for better understandings).

## Preparation
[Git](https://www.yiibai.com/git) | Github<br>
Ruby | Gem, Bundle...<br>
Jekyll | Markdown, Liquid, Rouge, [YAML](https://www.ibm.com/developerworks/cn/xml/x-cn-yamlintro/index.html)...<br>
*Optional: [RVM](https://ruby-china.org/wiki/rvm-guide), Disqus, Google Search Console...*

> As follows, I'm trying to describe in the briefest way. It's also good to get more detailed understandings [here](https://www.zhihu.com/question/30018945).

## Setup

### Github pages
1. Github > New Repository
2. Repo > Settings > GitHub Pages > Source: the branch that your site built from.
		
	* Your site is published at https://[username].github.io/[Repo_name]/

### Jekyll (local)
1. Ruby
2. RubyGems
3. Jekyll

```shell
gem install jekyll
```

## Operations

### Construct your site (code)
* Completely by yourself. Reference: [Jekyll Structure](https://www.jekyll.com.cn/docs/structure/)

	```shell
	git init
	```
* A template may help a lot, e.g., [Jekyll Themes](http://jekyllthemes.org/)

	```shell
	git clone <https://github.com/...> <folder>
	cd <folder>
	rm -rf .git
	git init
	```
	
### Update your contents*
```shell
git add (-A)
git commit -m <annotation>
```
### Test your blog locally
```shell
bundle exec jekyll serve
optinal: --watch, --drafts
```
### Publish your blog / posts*
```shell
git remote add origin git@github.com:<...>.git
git push -u origin <local branch>:<remote branch> (-f)
```
---
*\* A good editor may help a lot, e.g., VS Code :-)*


## Other Tips
### 1. Markdown
	A lightweight markup language.
	Syntax(CN DOC): https://markdown-zh.readthedocs.io/en/latest/

### 2. Liquid
	An template language written in Ruby.
	Syntax(CN DOC): https://liquid.bootcss.com/

### 3. [Hilighter (for jekyll)](https://blog.csdn.net/qiujuer/article/details/50419279)
* Rouge

		List of supported languages and lexers:
		https://github.com/jneen/rouge/wiki/List-of-supported-languages-and-lexers

## Some Issues
[解决Jekyll代码块无法正常显示Liquid代码问题](https://blog.csdn.net/JireRen/article/details/52197045)

	{ %  raw  % } ... { %  endraw  % }: temporarily disables tag processing

<br>

> Just for my own blog. So skip this part as you like.

## Cheat Sheet

<figure>
	<img src="{{site.url}}/images/myblog.png" alt="">
</figure>

### Navigation Links
To add additional links in the drop down menu edit `_data/navigation.yml`. (External links will open in a new window.)

### Truncate (Liquid)
`index.html`: to abstract each post, add

{% highlight liquid %}
{% raw %}
{{ post.content | truncatewords:150 }}
{% endraw %}
{% endhighlight %}

### Posts (YAML)

#### Layout
`_layouts/`<br>
1. page: `404.md`, `about.md`<br>
2. post-index: `index.html`, `tags/index.html`, `posts/index.html`<br>
3. post: `[xx-xx-xx]-[title].md`

#### Image
`feature`: inner background for the head.`_layouts/posts.html`<br>

* `credit`, `credit-link`: Image source
	
`entry`: shown on the index page.`index.html`
