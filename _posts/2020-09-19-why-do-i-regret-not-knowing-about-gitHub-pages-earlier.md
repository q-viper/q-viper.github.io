---
title:  "Why do I regret not knowing about GitHub Pages earlier?"
date:   2020-09-19 01:29:17 +0545
categories: github-pages
related_image: assets/scene.jpg
subtitle: A journal of Viper.
comments_id: 24
# permalink: personal/why-do-i-regret-not-knowing-about-gitHub-pages-earlier
---

**Contents**
* TOC
{:toc}

## Introduction
I am a CS Undergrad who always had a dream of writing a blogs and showcasing what I have to other people and motivate them. And no matter how hard I tried I failed everytime. The reason must be not to try more and fear of failing. It has been few months since I am blogging on my website [Ramkrishna Acharya](https://acharyaramkrishna.com.np). The domain .com.np is free for Nepalese. I write about some Machine Learning stuffs. I am using AWS's instance which is still free tier and I am using wordpress for the site maintaining. But the students and umemployed like me always have to search for the alternatives whenever our free tier runs out. I also made a portfolio using GitHub page but which is not that much great because it was only simple HTML and CSS of some public repository. Now I was fed up with the complexity of maintaining server for even a simple blogging site. Another problem I had was, I used to write blogs on Jupyter Notebook and copy the HTML content and paste on Wordpress blog as custom HTML. It used to work most of the time but sometimes I had to face errors like ** Can not update to database. ** So I searched about how can I do blogging with GitHub pages. Then within a few hours, I was able to write this blog (updated on September 25).

So I hope this blog will help you to start your own blogging site with the free and awesome GitHub pages and Jekyll.

## Credits
I am publishing this blog using mobile data which has been possible only by the support of [Vikram Krishna K](https://www.linkedin.com/in/vikram-krishna-k/). I want to give him huge credits for this.

## Create GitHub account
Of course we need one account. Everyone knows how to make one.

## Download and Install Ruby
In order to maintain our project locally, it is necessary to download *Ruby*. I hope you can find it on rubyinstaller.org/downloads for your system. I am using windows right now so I will write about it. 

**Follow**: [Klit&Code](https://www.kiltandcode.com/2020/04/30/how-to-create-a-blog-using-jekyll-and-github-pages-on-windows/) for more hints.

**Note:** Please read more about installation on rubyinstaller.org for your easy setup. 

## Make First Jekyll Project
* First install jekyll.
    Do `gem install jekyll bundler` from terminal.
* Check your version if necessary by using -v argument. i.e. `jekyll -v`
* From the terminal `jekyll new path-to-project`.
* To view initial project, `bundle exec jekyll serve` then goto `http://localhost:4000`.

## Deploy it on GitHub
There are 2 ways, easy but slower, hard but faster. The easier way is using GitHub on browser and harder is GIT. To learn about GIT you can always go to pages like [this one](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html). I am going after easy part.

### Make a Repository
* Go to your GitHub account and make a new repository under the name :`your-github-username.github.io`. For me, `q-viper.github.io`. 
* For easier setup, start with a Readme.md file.
* Then go to add files.
* Drag and drop the files you were working on earlier inside project folder.
* Then goto the new tab on browser and type `your-github-username.github.io` hit enter and TADAAAAA.

## Blogging
### Before writing a Math Blog
Just copy some files to your project from source theme. Type `bundle info minima` and hit enter to get onto minima directory. Goto that directory and copy the folders `_includes` and `_layouts` and paste them to project directory. 

#### On _config.yml
Add below code too.

`markdown: kramdown`<br>
`kramdown:`<br>
&ensp;`math_engine: mathjax`

[MathJax](https://www.mathjax.org) is awesome JS display engine that allows us to visualize Math Formulas. [Kramdown](https://kramdown.gettalong.org) according to developers is <i>kramdown (sic, not Kramdown or KramDown, just kramdown) is a free MIT-licensed Ruby library for parsing and converting a superset of Markdown. </i>

### Show Formula
Since I pretend to be Machine Learning guy, I have to write notations and symbols on blogs always so there is a better way of doing this by using MathJax. A Javascript Framework.

I searched and tried different solutions but none worked for me may be I was doing it on wrong way but finally I got through the post of [Ian Goodfellow](http://www.iangoodfellow.com/blog/jekyll/markdown/tex/2016/11/07/latex-in-markdown.html), father of GAN.

`<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>`

He suggested to use above code on `_includes/post.html` but seems putting it on top of `_includes/head.html` also works fine if we have formula on title too. It is not a magic but every blog page has stack of pages like header, footer etc.

### First Blog
* I hope you are a VS Code Fan. Goto the folder inside `_posts` of our project directory and create a new markdown file with the format like `2020-9-19-why-use-github-pages.md`.

* Then on the top of file, write somethinge like below:

    `---`<br>
    `layout: post `<br>
    `title:  "Why do I regret not knowing about GitHub Pages earlier?"`<br>
    `date:   2020-09-19 10:29:17 +0545`<br>
    `categories: jekyll update`<br>
    `---`

The above format is named as [front matter]https://jekyllrb.com/docs/front-matter/). The terms `layout`, `title` etc are known as variables. We can add our own varaibles to make more awesome blogs.
We can even use future date and the blog will be published on that date. The categories we used here will be automatically used on blog path by default. And the layout of this blog will be of post. It is interesting to see the file `_layouts/post.html`. We can only use layouts that are defined under that directory.
So the `$$ E=\frac{1}{2m} \sum_{i=0}^m {(y_i - t_i)^2} $$` is the LaTex code for RMS error on ML world.

$$ E=\frac{1}{2m} \sum_{i=0}^m {(y_i - t_i)^2} $$

Pretty Cool right?

## Make Awesome Posts
First thing we need is to make a custom category. Initially, we have only `jekyll`, you can see under the folder `_site`. But pretty thing is whenever we write a new post on markdown and give the category name different on the top bar(see above) then new category is made. But still the post will not be visible to the front page so what can we do is add a permalink on the `_config.yml`. i.e. `permalink: /:categories/:year/:month/:day/:title:output_ext`. Where `:categories` is placeholder for categories. Follow [this](https://jekyllrb.com/docs/permalinks/) official doc for more information.

## Add and View Comment
I scratched my head around for minutes and then I remembered some blogs where commenting was done by GitHub account. Jekyll is primarily used for static sites but comment falls under dynamic contents. There are various ideas though. Some people follows Disqus, and other stuffs. This [link](https://talk.jekyllrb.com/t/what-is-the-recommended-way-to-add-comment-sections-to-your-jekyll-blog/3330/6) contains a thread of commenting on Jekyll. And some people are smart to use GitHub's Issue as comment. Follow this [blog](https://aristath.github.io/blog/static-site-comments-using-github-issues-api) for more information. 

### GitHub Issue as Comment
If you visit my other blogs then you can see that you need GitHub account to comment and if you visit over my [repo](https://github.com/q-viper/q-viper.github.io/issues) you can see that there are open issues. One particular is on [this link](https://github.com/q-viper/q-viper.github.io/issues/24) which I am going to show here for commenting. The number 24 (not the movie Number 23) on the last is the number of issue. Before writing a blog it is essential to create a new app from [here](https://github.com/settings/applications/new). For filling out form, please see below image.
![png]({{site.url}}/assets/images/new_outh_app.png)

### Custom Comment File
To show our comments from GitHub issues, we have to make our new file on `_includes`. I am naming it `github_comments.html`. The content I am using is copied from this awesome [blog](https://www.aleksandrhovhannisyan.com/blog/dev/jekyll-comment-system-github-issues/). You can see entire code on [this link](https://github.com/q-viper/q-viper.github.io/blob/master/_includes/github_comments.html). 

### Assign Comment Id
Assign `comments_id` as variable on front matter. For every blog use one issue. And for this blog I am using [this](https://github.com/q-viper/q-viper.github.io/issues/24) issue. The 24 on the last is going to be our `comments_id`. Please see the file on this [link](https://raw.githubusercontent.com/q-viper/q-viper.github.io/master/_posts/2020-09-19-why-do-i-regret-not-knowing-about-gitHub-pages-earlier.md) for more information.

### Show Comment
Go to `_config.yml` and create the new site variable `issues_repo` and give its value as `your_github_username/your_issues_repo`. Now go inside `_layouts/post.htm` and on the part just after we do view content, do below
{% raw %}
    {% if page.comments_id %}
          {% include github_comments.html %}
    {% endif %} 
{% endraw %}

The liquid code above checks if the blog has the variable comments_id and if it does then show github_comments.html content. Now is the time to test it. Before viewing it, restart the server and then after some minutes the comments will be shown. just refresh it. Please check it below.

## Polish It
Sure the comments we are using right now is not that good. What we can do is add some CSS, JS and make it realtime commenting. But instead of that, I have used an awesome Jekyll theme [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/). I am using this same theme right now on this blog and I am more than gald to use it. I have seen many websites using this blog.