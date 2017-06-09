# KCF
========================
- 需要安装opencv，numpy，matplotlib
- 运行指令为`python run.py dataset_dir save_dir`，dataset_dir为图像序列所在目录，groundtruth文件也在该目录下，并命名为`groundtruth.txt`其bounding box的格式为左上角坐标及矩形框宽与高（x,y,w,h）,save_dir 为保存结果的目录。
- 其余程序参数可通过-h指令查看，目前scale功能还未实现
- 如当前目录下有文件夹为face的视频（文件夹内有图片序列及groundtruth.txt的文件），将结果保存于当前路径，其运行实例为： `python run.py face . `
![tracking example](example.png)
