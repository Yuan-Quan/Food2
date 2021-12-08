# -*- coding: utf-8 -*
from flyai.framework import FlyAI


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        pass

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/image\/Akarna_Dhanurasana35.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":1}
        '''
        return {"label": 1}
