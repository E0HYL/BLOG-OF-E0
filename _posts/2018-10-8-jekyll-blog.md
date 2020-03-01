---
layout: post
title: Jekyll + Github Pages = Blog
description: "Let me introduce how the blog works for my first post."
modified: 2020-3-1
tags: [Skills]
image:
  feature: abstract-1.jpg
---

> Though written in English, links here I attach are almost Chinese (for better understandings).

## Preparation
[Git](https://www.yiibai.com/git) | Github<br>
Ruby | Gem, Bundle...<br>
Jekyll | Markdown, Liquid, Rouge, [YAML](https://www.ibm.com/developerworks/cn/xml/x-cn-yamlintro/index.html)...<br>
*Optional: [RVM](https://ruby-china.org/wiki/rvm-guide), Disqus, Google Search Console...*

> As follows, I'm trying to explain in the briefest way. It's also good to get more detailed understandings [here](https://www.zhihu.com/question/30018945).

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

<!-- more -->

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
[`.gitignore` file doesn't work](https://www.jianshu.com/p/2b4222cc8734)

* Only ignore untracked files. Solution: delte all caches and commit then.

	```shell
	git rm -r --cached .
	git add .
	git commit -m 'update .gitignore'
	```

["Page Build Failure" from Github](https://github.com/mmistakes/so-simple-theme/issues/250)

* Your plugins won’t work if you’re deploying to GitHub Pages.
* To test locally, make sure you're using the GitHub Pages gem in your `Gemfile` and not Jekyll.
	
	```
	gem "github-pages", group: :jekyll_plugins
	```

[Emoji on GitHub Pages](https://help.github.com/articles/emoji-on-github-pages/#testing-locally)

* Add the plugin in `_config.yml`:

	```yaml
	plugins:
		- jemoji
	```

* To test locally, (delete Gemfile.lock), edit `Gemfile`, (run 'bundle update'):
	
	```
	gem 'gemoji'
	```

* Example: it's raining :cat:s and :dog:s! See [EMOJI CHEAT SHEET](https://www.webpagefx.com/tools/emoji-cheat-sheet/) for more.

[Kramdown: Inser Markdown into HTML](https://ask.helplib.com/html/post_995628)

* If an HTML tag has an [attribute markdown="1"](https://kramdown.gettalong.org/syntax.html#html-blocks), then the default mechanism for parsing syntax in this tag is used.
	
	```html
	<div markdown="1">
	My text with **markdown** syntax
	</div>
	```

[Liquid code can't be displayed normally](https://blog.csdn.net/JireRen/article/details/52197045)

* {% raw %}{% raw %}{% endraw %}{% raw %}\{% endraw %\}{% endraw %}: temporarily disables tag processing

<br>

> Just for my own blog. So skip this part as you like.

## Cheat Sheet

<figure>
	<img src="{{site.url}}/images/myblog.png" alt="">
</figure>

### Posts (YAML)

#### Layout
`_layouts/`<br>
1. page: `404.md`, `about.md`<br>
2. post-index: `index.html`, `tags/index.html`, `posts/index.html`<br>
3. post: `_posts/[xx-xx-xx]-[title].md`

#### Image
`feature`: inner background for the head.`_layouts/posts.html`<br>

* `credit`, `credit-link`: Image source
	
`entry`: shown on the index page.`index.html`

### Navigation Links
To add additional links in the drop down menu edit `_data/navigation.yml`. (External links will open in a new window.)

### ~~Truncate (Liquid)~~ (use excerpts instead)
`index.html`: To abstract each post, add
```liquid
{% raw %}
{{ post.content | truncatewords:150 }}
{% endraw %}
```

### Post excerpts
- use `excerpt` variable on a post (default: the first paragraph of content).
- customized by setting a `excerpt_separator` variable in YAML front-matter or _config.yml.
	```yml
	---
	excerpt_separator: <!--more-->
	---

	Excerpt
	<!--more-->
	Out-of-excerpt
	```

### Support MathJax
`_layouts/post.html`: Add script in <head>
```javascript
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```
More usages [here](http://jzqt.github.io/2015/06/30/Markdown%E4%B8%AD%E5%86%99%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F/).
