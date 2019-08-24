## KDD‘19 basic implementation code

This is basic implementation of our KDD'19 paper:

## Requirements
- Ubuntu 16.04
- Python 3.6
- Tensorflow-gpu
- GPU,CUDA,CUDNN

Note: Running this project will consume upwards of 100GB hard disk space. The overall pipeline will take several hours. You are recommended to run this project on a Linux server.

## Code Description
our model is placed in the _global__ directory. Some auxiliary programs are placed in _utils_ directory.
Specifically, In the _global__ directory, global_model.py is our ranking model, gen_train_data.py is the data processer of our ranking model. cl_test.py is our classification model which import pre_trained ranking model to train and jointly train. classifier_data.py is the data processer of our classification model. classifier_settings.py is the hyper-para setting of classification model.

## Data Source
All experimental data is available.(link: https://www.aminer.cn/na-data)

Note: Due to the data file which should be added to run the code is too large, (e.g. embedding words file, the per-trained checkpoints file,etc) we just upload our basic code now. After we have carefully sorted out the code and data, we will upload the complete version again.
