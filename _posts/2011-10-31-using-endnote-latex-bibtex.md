---
layout: post
title:  "使用 EndNote 和 LaTeX/BibTeX 建立论文的参考文献"
date:   2011-10-31 21:17:59 +0800
categories: 
- Notes 
tags:
- EndNote
- LaTeX

---

[EndNote](http://www.endnote.com/) 是整理论文参考文献 (bibliography) 的绝佳工具。如果能把 EndNote 与 LaTeX 结合使用是再好不过了。这里介绍一种将 EndNote 内的书目列表导出为 LaTeX 可用的 BibTeX 格式的方法。 

这里使用的软件版本为 EndNote X6。NTU 的学生可以在[这里](http://www.ntu.edu.sg/Library/Lip/endnote/Pages/download.aspx)下载最新版本免费使用。假设 EndNote 里已经添加了所需的参考文献信息。很多数据库都自带导入 citation 到 EndNote 的功能，这里就不详细介绍。

## 一、EndNote 的配置
---

* 下载[这个文件]({{ site.url }}/assets/downloads/IEEE BibTeX Export.ens)，然后把它放到 `%Userprofile%\Documents\EndNote\Styles` 文件夹中 (如果没有该文件夹则自行创建)。我这里使用的是 IEEETrans 兼容的格式，如果需要其他格式，得修改 style template。
* 打开 EndNote，从菜单栏里选取 **Edit -> Output Styles -> Open Style Manager**
* 找到并勾选第一步中的 IEEE BibTeX Export，关闭该窗口
* 从菜单栏中 **Edit -> Output Styles** 选刚增加的 IEEE BibTeX Export
* 编辑条目：双击需要引用的参考资料，在 Label 一栏里填一个唯一标识符。我的 label 命名习惯是 "第一作者姓氏（或全名）+发表年份+一个字母"。最后的字母是从a往后递增，为了区别同一作者在同一年发的不同文章。比如 Danny Li 在 2011 年发的第一篇文章，label 就是 dannyli2011a。
* 在 EndNote 主窗口中选中所有需要引用的参考资料，菜单 **File -> Export**，Output Style 选 IEEE Trans，保存为一个 txt 文档到 LaTeX 的工作目录中。然后更改文件后缀名为 .bib，如 dannybib.bib；

## 二、CTeX 中使用 BibTeX
---

建立好 .bib 文件后就可以按照常规使用 BibTeX 的方式建立文献参考了。

*  编写 LaTeX 代码；
*  在结尾前加入 BibTeX 相关代码，如 

		
~~~ latex
\bibliographystyle{plain}
\bibliography{dannybib} % dannybib是Endnote导出的文件名
~~~

*  正文中需要引用的地方就可以加入

~~~ latex
\cite{dannyli2011a} % dannyli2011a 是之前在Endnote 中设置的 label
~~~

## 三、可能遇到的问题
---

EndNote 导出的 BibTeX 文件可能不能直接使用:

*  书目名有特殊符号的，需要在前面加反斜杠 "\\"；
*  书目名字母要强行大写或小写，要在两边加花括号 { }；
*  关于每条 citation 的样式，是由 BSTcontrol 设置的。这里要手动在导出的 .bib 文件最后添加以下代码

~~~ latex
@IEEEtranBSTCTL{BSTcontrol,
  CTLuse_article_number     = "yes",
  CTLuse_paper              = "yes",
  CTLuse_forced_etal        = "yes",
  CTLmax_names_forced_etal  = "2",
  CTLnames_show_etal        = "1",
  CTLuse_alt_spacing        = "yes",
  CTLalt_stretch_factor     = "4",
  CTLdash_repeated_names    = "yes",
  CTLname_format_string     = "{f.~}{vv~}{ll}{, jj}",
  CTLname_latex_cmd         = "",
  CTLname_url_prefix        = "[Online]. Available:"
}
~~~

## References
---
*   Bevan S. Weir, [Step-by-step guide to using EndNote with LaTeX and BibTeX](http://www.rhizobia.co.nz/latex/convert.html)
*   Michael Shell, [How to Use the IEEEtran BIBTEX Style](http://ctan.unixbrain.com/macros/latex/contrib/IEEEtran/bibtex/IEEEtran_bst_HOWTO.pdf)