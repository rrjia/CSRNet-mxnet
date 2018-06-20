# CSRNet-mxnet
CVPR CSRNet Mxnet implement

##
实验采用TRANCOS_v3数据 http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/

使用data_process.py处理原始数据，生成mask之后的image作为输入，根据txt文件夹中的文件生成densitymap 作为ground truth,作为标签。
然后使用CSRnet中代码进行训练评估以及对自己的数据进行密度和数量统计。

训练模型
train_model()
评估模型
eval_model()
评估自己的数据
predict_dir_image_car_num('dir_name')
