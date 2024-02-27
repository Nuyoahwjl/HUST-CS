该模板是基于https://github.com/skinaze/HUSTPaperTemp  项目修订
用法大概就是写好了敲make就行。
windows环境下装个miktex然后想办法随便从哪找个make就可以

来自jzchen的补充
1. 建议装textstudio编辑, 因为标记比较明显. 但不能用它编译
2. 需要装texlive, 可以在线装也可以下载镜像装, 镜像有4G (https://mirrors.sustech.edu.cn/CTAN/systems/texlive/Images/texlive.iso)
3. 不建议用Texworks编译, 界面不友好, 标记不明显, 搜索不方便, 不能生成参考文献
4. 解决了不能直接生成并引用参考文献的问题
5. 减小了目录中标题到页码的指引线(虚线点之间)的间距
6. 调整了封面论文题目的宽度, 题目栏一行能放下不多于19个汉字. 建议题目字数少于19
7. 正文行距从1.62调整到1.5倍
8. 补充了更多的类型的bib格式
9. 章末尾增加了\newpage强行分页
10. 采用了另外一种插图方式
11. 增加了公式示例
12. 增加了makethesis.bat与makethesis文件
13. 如果想在textstudio中配编译环境请看: https://zhuanlan.zhihu.com/p/138586028?utm_source=qq&utm_medium=social&utm_oi=1192103921202638848, 但老师还是建议用cmd
14. 插图用visio画, 打印成成pdf文件, 如1-1.pdf, 3-2.pdf, 并裁剪去掉周边空白

编译方法:
1. 在当前文件夹的资源管理器的地址栏中输入cmd再回车, 切记切记编译时候必须把HustGraduPaper.pdf文件关掉, 如果忘记关了, 造成编译时出错, 可以试着删除bbl文件
2. 输入makethesis并回车

advices from jzchen:
插图建议用visio画, 然后打印成pdf, 再用adobe pdf自带的裁剪工具裁剪, 再在文中插入该pdf文件. 则生成与插入的为矢量图效果好.
