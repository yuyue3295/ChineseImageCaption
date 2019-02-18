# ChineseImageCaption
自动描述图像中内容是人工智能最基本的问题，它连接了计算机视觉和自然语言处理。图像描述必须捕获图像中的内容，还必须表达清楚图像中物体之间的描述.
flickr30k图像数据集以及图像中英文的文本描述链接是：https://pan.baidu.com/s/1FNql2UJ4QX9N_9bdTExiaQ ，其中data.zip文件包含flickr30k_images文件夹，进行过简单清洗过后的图像的中文文本描述clean_zh_results_20130124.token，图像的英文文本描述results_20130124.token，以及单词统计信息zh_vocab.txt等等。<br>
模型文件的连接是：https://pan.baidu.com/s/1z3YYGqA2ewgqe5lf_0yYFA ，其中model_files.zip包含feature_extraction_inception_v3文件夹，里面是提取到的图像的transfer values的pkl文件；inception_v3 文件夹，里面是inception_v3_graph_def.pb；local_run文件，里面是训练了100轮共31799步的模型文件等。代码运行请参照 项目文件目录.JPG中的目录放置解压的文件<br>
测试图像生成文本模型的效果请在jupyter notebook中运行image_caption_test.ipynb 文件。<br>
![](img/inception_v3.png)

