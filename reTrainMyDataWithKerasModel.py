from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
import timeit
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("本py用来构建在老模型的基础上再训练。")



### 超参数：
# dimensions of InceptionV3.
old_model_path = "data/modelFile/my_model02.h5" # 你要在哪个模型的基础上训练？
new_model_path = "data/modelFile/my_model03.h5" # 新模型存储到哪里？
img_width, img_height = 224, 224 #
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 8571
nb_validation_samples = 80
epochs = 10 # 图像的轮数
batch_size = 32 # 每个batch32张图片
learning_rate = 0.0001 # 如果在原有图片的基础上再训练，为防治过拟合，建议设置小一点：0.0001
                       # 如果是新的图片数据，且比较重要，建议设置:0.001 甚至更高：0.002


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

### 返回一个编译好的模型
print("开始加载模型：")
starttime = timeit.default_timer()
model = load_model(old_model_path)
endtime = timeit.default_timer()
print("模型加载完成，用时为：", endtime - starttime)

# 让我们可视化层名称和层索引以查看有多少层
print("打印old模型结构：")
for i, layer in enumerate(model.layers):
   print(i, layer.name) # 311层 + 2 dense

# 我们选择训练 top 2 个dense块，即我们将冻结前311层(锁住所有 InceptionV3 的卷积层)，其余的解冻：
for layer in model.layers[:-2]:
   layer.trainable = False
for layer in model.layers[-2:]:
   layer.trainable = True

## 我们需要重新编译模型以使这些修改生效
# 注意:（一定要在锁层以后操作）
# 我们使用的学习率较低，微调
model.compile(optimizer=RMSprop(lr = learning_rate), loss='categorical_crossentropy',metrics=["accuracy"])

### 数据集
# 这是我们将用于训练的扩充配置
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2, # 用来进行剪切变换的程度
    zoom_range=0.2, # 用来进行随机的放大
    horizontal_flip=True) # 随机的对图片进行水平翻转
# 这是我们将用于测试的扩充配置：
valid_datagen = ImageDataGenerator(rescale=1. / 255)# 仅缩放

# 这是一个生成器，将读取在“train_data_dir”子文档中找到的图片，并无限期地生成一批增强图像数据
train_generator = train_datagen.flow_from_directory( # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签。
    train_data_dir,
    target_size=(img_width, img_height), # 所有图像将调整为224x224
    batch_size=batch_size,
    class_mode='categorical',
    # classes = ["neutral","political","porn","terrorism"]
)
print(train_generator.class_indices) #{'neutral': 0, 'political': 1, 'porn': 2, 'terrorism': 3}


validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    # classes = ["neutral","political","porn","terrorism"]
)
print(validation_generator.class_indices) # {'neutral': 0, 'political': 1, 'porn': 2, 'terrorism': 3}

# 在新的数据集上训练几代
print("开始训练：")
model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size, # 一个 epoch 完成并开始下一个 epoch 之前,如果未指定，将使用len(generator) 作为步数。
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

print("开始validation")


## 保存模型
model.save(new_model_path)  # 创建 HDF5 文件 'my_model01.h5'
print("保存模型成功。")


### 使用模型预测：
print("使用模型预测：-----------")
img_path = 'data/validation/neutral/[www.google.com][1989].jpg' # neutral 2
# 加载图像
img = image.load_img(img_path, target_size = input_shape)
# 图像预处理
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 对图像进行分类
preds = model.predict(x) # Predicted: [[1.0000000e+00 1.4072199e-33 1.0080164e-22 3.4663230e-32]]
classes = np.argmax(preds,axis=1)
# 输出预测概率
print('Predicted:', preds)
print('classes:', classes)
