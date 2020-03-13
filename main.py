# -*- coding: UTF-8 -*-
import json
import logging
import timeit
import flask as flask
import predictWithMyModel

##### flask初始化
ImageIllehalDetectionApp = flask.Flask(__name__)  # __name__代表当前的python文件。把当前的python文件当做一个服务启动


#### Hello World !
@ImageIllehalDetectionApp.route("/helloWorld")
def helloWorld():
    """
    当用户访问这个端口时，返回Hello World !
    :return:
    """
    return "图像内容违规检测，涉黄、涉政、涉恐，Hello World ! "


#### 图片检测的接口
@ImageIllehalDetectionApp.route("/image/detection", methods=['post'])
def imageDetection():
    """
    图片检测的接口
    """
    imageBase64String = flask.request.values.get('imageBase64String')# 接收post参数
    if not imageBase64String :
        # 传入参数为空的错误控制
        ret = {}
        ret['code'] = "98"
        ret['message'] = "缺少必要传入参数"
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)


    startTime = timeit.default_timer()  # 记录开始时间
    code, msg, predictedClass, probability = predictWithMyModel.predictWithImageBase64(imageBase64String) # 查询数据
    endTime = timeit.default_timer()  # 记录 结束时间
    print('-------------------  flask图片检测时间：', endTime - startTime, 's -------------------')
    logger.info('------------------- flask图片检测时间：{}s ------------------- '.format(endTime - startTime))

    # 定义flask返回值ret的数据结构
    if code == '00':
        ret = {}
        ret['code'] = code
        if int(probability) < 0.75:
            predictedClass = "neutral" # 若盖里比较低说明，模型判断不准，就返回无违规
        ret['message'] = '图片检测成功，类别是：'+ str(predictedClass)
        ret['predictedClass'] = predictedClass
        ret['probability'] = str(probability)
        print(ret)
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)
    elif code == '98':
        ret = {}
        ret['code'] = code
        ret['message'] = '图片检测失败，原因是：'+str(msg)
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)





if __name__ == '__main__':
    # 配置日志
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./log/log_my_image.log', level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')


    ##### flask初始化
    ImageIllehalDetectionApp.run(
        port=8133,
        debug=False,
        threaded=False,
        host='0.0.0.0'
    )

