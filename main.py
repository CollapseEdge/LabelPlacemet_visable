import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import networkx as nx
import pandas as pd
from tqdm import tqdm

class DataSets():
    def __init__(self,FULL_DATA_SETS_PATH,PositionNP,category_csv,pred_npy_path,todo_list):
        self.PURE_PNG_PATH =    FULL_DATA_SETS_PATH + '/PNGImage-P'
        self.ANCHOR_NPY_PATH =  FULL_DATA_SETS_PATH + '/' + PositionNP
        self.JSON_PATH =        FULL_DATA_SETS_PATH + '/ImageAnnotation'
        self.LAYOUT_PATH =      FULL_DATA_SETS_PATH + '/Layout/' + todo_list
        self.CATEGORY_PATH =    FULL_DATA_SETS_PATH + '/' + category_csv
        self.PRED_NPY_PATH =    pred_npy_path
        
    def read_png(self, png_path):
        png_data = cv2.imread(png_path)
        return png_data

    def read_npy(self, npy_path):
        npy_data = np.load(npy_path)
        return npy_data
    
    def read_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data
    
    def read_layout(self):
        layout_data = []
        with open(self.LAYOUT_PATH, 'r') as f:
            for line in f:
                layout_data.append(line.strip())
        return layout_data
        
    def find_category(self, id):
        data = pd.read_csv(self.CATEGORY_PATH)
        primary_data = id + '.json'
        for category in data.columns:
            if primary_data in data[category].values:
                row_index = data[data[category] == primary_data].index[0]
                return category
            
    def get_pred_npy(self, id):
        pred_npy = np.load(self.PRED_NPY_PATH + '/' + id + '.npy')
        return pred_npy
    
    def get_full_path(self, id):
        category =  self.find_category(id)
        png_path =  self.PURE_PNG_PATH + '/' + category + '/' + id + '-P.png'
        npy_path =  self.ANCHOR_NPY_PATH + '/' + category + '/' + id + '.npy'
        json_path = self.JSON_PATH + '/' + category + '/' + id + '.json'
        #print(png_path, npy_path, json_path)
        return png_path, npy_path, json_path
    
    def get_data_in_json(self, json_data):
        Height =    json_data['imageHeight']
        Width =     json_data['imageWidth']
        text_shapes = []
        for shape in json_data['shapes']:
            if shape['label'] == 'Text':
                text_shapes.append(shape)
        
        rectrangle_info = []
        
        for label in text_shapes:
            group_id =              label['group_id']
            points =                label['points']
            rectrangle =            [(points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2]
            rectrangle_height =     np.abs(points[1][1] - points[0][1])
            rectrangle_width =      np.abs(points[1][0] - points[0][0])
            info_dict =             {'group_id': group_id, 'rectrangle': rectrangle, 'rectrangle_height': rectrangle_height, 'rectrangle_width': rectrangle_width}
            rectrangle_info.append(info_dict)
            
        return Height, Width, rectrangle_info
    
    def denormalization(self, data, Width, Height):
        for i in range(len(data)):
            data[i][0] = data[i][0] * Width
            data[i][1] = data[i][1] * Height
        return data

class Drawer():
    def __init__(self, height, width, png_data, npy_data, rectrangle_info, pred_npy_denormalization, output_path, id):
        #生成颜色列表
        cmap = plt.get_cmap('tab20')
        colors = list()
        for c in range(40):
            r = cmap(c)[0]
            g = cmap(c)[1]
            b = cmap(c)[2]
            colors.append((r, g, b))
            
        self.height =           height
        self.width =            width
        self.png_data =         png_data
        self.npy_data =         npy_data
        self.rectrangle_info =  rectrangle_info
        self.pred_npy =         pred_npy_denormalization
        self.colors =           colors
        self.output_path =      output_path
        self.id =               id

    def draw_text():# TODO
        pass
        
    def plot(self, run_model, left_center, debug, dpi):
        
        # 设置画布大小
        plt.figure(figsize=(self.width / 100, self.height / 100))
        line_width = 0.001*min(self.height,self.width)
        
        if len(self.pred_npy) != len(self.rectrangle_info):
            print("Error: pred_npy and rectrangle_info length not equal!")
            return False
        # 将self.rectrangle_info中的的数据按照group_id进行排序
        self.rectrangle_info = sorted(self.rectrangle_info, key=lambda x: x['group_id'])
        if debug:
            print(self.width, self.height)
            print('预测的Text框信息：',end='\n')
            print(self.pred_npy)
            print('anchor点位信息：',end='\n')
            print(self.npy_data)
            print('ground-truth的Text框信息：',end='\n')
            print(self.rectrangle_info)
        
        for i in range(len(self.rectrangle_info)):
            ''' 预测的文本框的中心点暂时不需要
            if run_model == 'plot':
                # 绘制预测文本框的中心点
                if left_center == 1:
                    plt.scatter(self.pred_npy[i][0], self.pred_npy[i][1], s=5, color=self.colors[i], alpha=0.5)# 当传入的数据是中心点时，可以用这个来画点
                elif left_center == 0: #TODO 这有个bug要修理
                    plt.scatter(self.pred_npy[i][0] + int(self.rectrangle_info[i]['rectrangle_width']) / 2, self.pred_npy[i][1] + int(self.rectrangle_info[i]['rectrangle_height']), s=5, color=self.colors[i], alpha=0.5)# 当传入的数据是左上角点时，可以用这个来画点
            elif run_model == 'study' or run_model == 'Text':
                pass
            '''

            # 绘制ground truth的anchor的中心点
            #print(self.rectrangle_info)
            plt.scatter(self.npy_data[i][0], self.npy_data[i][1], s=5, color=self.colors[i], alpha=1)
            
            # 从 self.rectrangle_info[i] 中获取矩形的信息
            info = self.rectrangle_info[i]
            center_x, center_y =    info['rectrangle']
            width =                 info['rectrangle_width']
            height =                info['rectrangle_height']
            x =                     center_x - width / 2
            y =                     center_y - height / 2

            if run_model == 'plot' or run_model == 'study':
                #绘制anchor与预测Text框的中心点的连线
                if left_center == 1:
                    plt.plot([self.pred_npy[i][0], self.npy_data[i][0]], [self.pred_npy[i][1], self.npy_data[i][1]], color=self.colors[i])# 当传入的数据是中心点时，可以用这个来画线
                elif left_center == 0:
                    plt.plot([self.pred_npy[i][0] + self.rectrangle_info[i]['rectrangle_width'] / 2, self.npy_data[i][0]], [self.pred_npy[i][1] + self.rectrangle_info[i]['rectrangle_height'] / 2, self.npy_data[i][1]], color=self.colors[i]) # 当传入的点是左上角点时，可以用这个来画线
            elif run_model == 'Text':
                #绘制anchor与真实Text框的中心点的连线
                plt.plot([center_x, self.npy_data[i][0]], [center_y, self.npy_data[i][1]], color=self.colors[i])

            # plot模式，都画
            if run_model == 'plot':
                '''创建ground truth的矩形框'''
                rect = pch.Rectangle((x, y), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none')
                plt.gca().add_patch(rect)

                '''绘制pred矩形框'''
                '''这里有两种npy的储存方式，一种是中心点，一种是左上角点，这里根据实际情况选择一种即可'''
                if left_center == 1:
                    rect_pred = pch.Rectangle((self.pred_npy[i][0] - width / 2, self.pred_npy[i][1] - height / 2), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none', linestyle='--') # 中心点
                elif left_center == 0:
                    rect_pred = pch.Rectangle((self.pred_npy[i][0], self.pred_npy[i][1]), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none', linestyle='--') # 左上角点

                plt.gca().add_patch(rect_pred)

            # study模式，只画pred
            if run_model == 'study':
                '''绘制pred矩形框'''
                '''这里有两种npy的储存方式，一种是中心点，一种是左上角点，这里根据实际情况选择一种即可'''
                if left_center == 1:
                    rect_pred = pch.Rectangle((self.pred_npy[i][0] - width / 2, self.pred_npy[i][1] - height / 2), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none', linestyle='--') # 中心点
                elif left_center == 0:
                    rect_pred = pch.Rectangle((self.pred_npy[i][0], self.pred_npy[i][1]), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none', linestyle='--') # 左上角点

                plt.gca().add_patch(rect_pred)

            # Text模式，只画ground-truth
            if run_model == 'Text':
                '''创建ground truth的矩形框'''
                rect = pch.Rectangle((x, y), width, height, linewidth=line_width, edgecolor=self.colors[i], facecolor='none')
                plt.gca().add_patch(rect)
                #self.draw_text()
        
        plt.imshow(self.png_data)
        # 取消白边
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # 添加黑色边框
        #ax = plt.gca()
        #ax.add_patch(plt.Rectangle((0, 0), self.width, self.height, color='black', linewidth=3, fill=False)) # //TODO //FIXME 这个黑边这次先不加了，画出来粗细不一，不好看
        if debug:
            plt.show()
        else:
            # 保存图片
            plt.savefig(self.output_path + '/' + self.id +'.png', dpi=dpi)
            plt.close()

def main(options):
    data_sets_path =    options['datasets_path']
    npy_path =          options['postionNP_path']
    category_csv =      options['category_csv']
    pred_npy_path =     options['pred_data']
    output_path =       options['output_path']
    denormalization =   options['denormalization']
    task_list =         options['task_list']
    left_center =       options['left_or_center']
    debug =             options['debug']
    dpi =               options['dpi']
    run_model =         options['run_model']


    # 检测output_path是否存在，不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    FULL_DATA_SETS_PATH = data_sets_path
    dataloader = DataSets(FULL_DATA_SETS_PATH, npy_path, category_csv, pred_npy_path, task_list)
    test_data = dataloader.read_layout()
    for id_long in tqdm(test_data, desc='Processing', unit='items', leave=True, dynamic_ncols=True):
        id = id_long.split(':')[0]
        png_path, npy_path, json_path = dataloader.get_full_path(id)
        try:
            png_data =  dataloader.read_png(png_path)
            npy_data =  dataloader.read_npy(npy_path)
            json_data = dataloader.read_json(json_path)
            pred_npy =  dataloader.get_pred_npy(id)
            #print(pred_npy)
        except:
            print(f"Error: {id} data read error!")
            continue 
        # 判断是png_data，npy_data，json_data是否读取成功
        if png_data is None or npy_data is None or json_data is None:
            print(f"Error: {id} data is None!")
            continue
        height,width,rectrangle_info = dataloader.get_data_in_json(json_data)
        # 这里选择反归一化或者不进行归一化
        if denormalization:
            pred_npy_denormalization = dataloader.denormalization(pred_npy, width, height) # 反归一化
        else:
            pred_npy_denormalization = pred_npy # 不进行归一化
        drawer = Drawer(height, width, png_data, npy_data, rectrangle_info, pred_npy_denormalization,output_path, id)
        drawer.plot(run_model, left_center, debug, dpi)
        
        # 销毁对象, 释放内存
        del drawer
        del png_data
        del npy_data
        del json_data
        del pred_npy
        del pred_npy_denormalization
        del rectrangle_info
        del height
        del width
        
        if debug:
            break
            #pass

    del dataloader

if __name__ == '__main__':
    options = {
        'denormalization':      True,                               #传入的数据是否需要反归一化, bool
        'run_model':            'Text',                            #选择运行模型。Text是画有文字内容的图片，plot是绘制ground-truth和pred的对比图，study是绘制userstudy用的图片 ['plot','study','Text']
        'dpi':                  300,                                #保存图像时用的dpi, int
        'left_or_center':       1,                                  #0是左上角，1是中心点 [0,1]
        'datasets_path':        'G:\SWUIllustration_869_v1',        #数据集的位置 string
        'postionNP_path':       'PositionNP_new2',                  #npy格式的anchor文件夹的位置 string
        'category_csv':         'category-869.csv',                 #存放类别信息的csv string
        'pred_data':            './LPGT',                           #预测的Text框，以npy的格式储存 string
        'task_list':            'new.txt',                          #要进行可视化的图片，放在Layout文件夹下 string
        'output_path':          './lpgt_output_new1',               #输出图片的存放位置 string
        'debug':                True                               #是否以debug模式运行,只画一张图，并且显示图片 bool
    }

    main(options)
    