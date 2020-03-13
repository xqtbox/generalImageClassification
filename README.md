# generalImageClassification
[toc]

本项目是通用的图像分类项目，并以涉黄、涉政、涉恐和普通图片4分类为例。图像违规质检本质是图像分类，所以关键点在于两个：

1. 图像分类的数据准备；
2. 图像分类的模型选择、训练；

## 1 数据准备

为了达到特定类别的分类，准备相应的图片数据，
- 1 开源的数据集。
- 2 自己写爬虫，爬取数据。但是没时间写，而且反爬虫设施的破解很费时间。
- 3 利用特定的网站（爬虫），帮你取下载数据。

### 1.1 开源数据集

如果开始一个图像相关的项目，而这个领域又有公开、开源的数据集，那是最幸福的一件事了。所以有了项目需求之后，第一件事情，可以去github等网站搜寻一下有没有可以直接使用的数据集。

而对我们的“图片质检”项目，涉政图片、涉恐图片网上找不到现成的数据集。但是涉黄图片却又很多公开数据集，并且图片质量灰常的“优秀”。下面给出两个实例：

1. nsfw_data_scrapper公开数据集（下面是图片地址，和一些介绍如何使用的博客）：
    - nsfw_data_scraper 数据 https://github.com/alex000kim/nsfw_data_scraper
    - NSFW Model（使用nsfw_data_scrapper数据训练resnet） https://github.com/rockyzhengwu/nsfw
    - nsfw_data_scraper 博客 https://www.ctolib.com/topics-137790.html
    - nsfw_data_scraper 博客 https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/86653609
2. nsfw_data_source_urls公开数据集：
    - 另外一个数据库nsfw_data_source_urls： https://github.com/EBazarov/nsfw_data_source_urls
    - nsfw_data_source_urls：博客https://www.tinymind.cn/articles/4025




### 1.2 利用特定网站爬数据
最近发现一个特别好的图片爬虫网站：[imagecyborg](https://imagecyborg.com/)，只需要把你想下载的图片的网页地址放进去，他就可以帮你打包下载。使用方法：
- 谷歌图像搜索相关的关键词。
- 把网址放入 https://imagecyborg.com/ 中帮你下载。

## 2 分类模型的选择

图像分类研究近些年已经非常的成熟，涌现出的大批的优秀模型,并且已经被深度学习框架纳入自己的版图.对于工程界的我们一般只需要微调这些成熟的模型即可。

下面是keras框架内置的模型介绍，本项目考虑准确率和运行速度两个问题，所以选择了InceptionV3这个模型，准确率也可以，模型也不太深 参数不算过多。
|模型|大小|Top1准确率|Top5准确率|参数数量|深度|
|:-:|:-:|:-:|:-:|:-:|:-:|
Xception	|88 MB	|0.790	|0.945	|22,910,480	|126
InceptionV3	|92 MB	|0.779	|0.937	|23,851,784	|159
ResNeXt50	|96 MB	|0.777	|0.938	|25,097,128	|-
DenseNet201	|80 MB	|0.773	|0.936	|20,242,984	|201
DenseNet169	|57 MB	|0.762	|0.932	|14,307,880	|169
ResNet50V2	|98 MB	|0.760	|0.930	|25,613,800	|-
DenseNet121	|33 MB	|0.750	|0.923	|8,062,504	|121


> 参考：
    - keras微调模型  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    - 《Keras：自建数据集图像分类的模型训练、保存与恢复》 https://blog.csdn.net/akadiao/article/details/80456742

## 3 代码结构及使用方法

### 3.1 代码结构
- data：存放数据的文件夹。
    - train：存放训练图片数据的文件夹。
    - validation：存放validation图片数据的文件夹。
    - modelFile：存放训练好的模型的文件夹。
- log：存放log的文件夹。
- trainMyDataWithKerasModel.py：训练模型的文件。
- predictWithMyModel.py：使用训练好的模型进行单次预测。
- reTrainMyDataWithKerasModel.py：对训练好的模型进行再训练。
- main.py：使用flask将predictWithMyModel.py作为接口开放出去。



### 3.2 使用方法

主要需求的python包：
- tensorflow：1.4.0
- keras：2.3.1
- flask：1.0.2 （非必须）

数据输入 和 数据lable生成：项目使用了keras的ImageDataGenerator这一个神器，会根据data文件夹下的子文件夹的名字生成lable。