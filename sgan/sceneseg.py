import torch
import numpy as np
import cv2
import math

def label_name(i):
      obj = np.array(["wall",
            "building, edifice",
            "sky",
            "floor, flooring",
            "tree",
            "ceiling",
            "road, route",
            "bed",
            "windowpane, window",
            "grass",
            "cabinet",
            "sidewalk, pavement",
            "person, individual, someone, somebody, mortal, soul",
            "earth, ground",
            "door, double door",
            "table",
            "mountain, mount",
            "plant, flora, plant life",
            "curtain, drape, drapery, mantle, pall",
            "chair",
            "car, auto, automobile, machine, motorcar",
            "water",
            "painting, picture",
            "sofa, couch, lounge",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug, carpet, carpeting",
            "field",
            "armchair",
            "seat",
            "fence, fencing",
            "desk",
            "rock, stone",
            "wardrobe, closet, press",
            "lamp",
            "bathtub, bathing tub, bath, tub",
            "railing, rail",
            "cushion",
            "base, pedestal, stand",
            "box",
            "column, pillar",
            "signboard, sign",
            "chest of drawers, chest, bureau, dresser",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace, hearth, open fireplace",
            "refrigerator, icebox",
            "grandstand, covered stand",
            "path",
            "stairs, steps",
            "runway",
            "case, display case, showcase, vitrine",
            "pool table, billiard table, snooker table",
            "pillow",
            "screen door, screen",
            "stairway, staircase",
            "river",
            "bridge, span",
            "bookcase",
            "blind, screen",
            "coffee table, cocktail table",
            "toilet, can, commode, crapper, pot, potty, stool, throne",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove, kitchen stove, range, kitchen range, cooking stove",
            "palm, palm tree",
            "kitchen island",
            "computer, computing machine, computing device, data processor, electronic computer, information processing system",
            "swivel chair",
            "boat",
            "bar",
            "arcade machine",
            "hovel, hut, hutch, shack, shanty",
            "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
            "towel",
            "light, light source",
            "truck, motortruck",
            "tower",
            "chandelier, pendant, pendent",
            "awning, sunshade, sunblind",
            "streetlight, street lamp",
            "booth, cubicle, stall, kiosk",
            "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
            "airplane, aeroplane, plane",
            "dirt track",
            "apparel, wearing apparel, dress, clothes",
            "pole",
            "land, ground, soil",
            "bannister, banister, balustrade, balusters, handrail",
            "escalator, moving staircase, moving stairway",
            "ottoman, pouf, pouffe, puff, hassock",
            "bottle",
            "buffet, counter, sideboard",
            "poster, posting, placard, notice, bill, card",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
            "canopy",
            "washer, automatic washer, washing machine",
            "plaything, toy",
            "swimming pool, swimming bath, natatorium",
            "stool",
            "barrel, cask",
            "basket, handbasket",
            "waterfall, falls",
            "tent, collapsible shelter",
            "bag",
            "minibike, motorbike",
            "cradle",
            "oven",
            "ball",
            "food, solid food",
            "step, stair",
            "tank, storage tank",
            "trade name, brand name, brand, marque",
            "microwave, microwave oven",
            "pot, flowerpot",
            "animal, animate being, beast, brute, creature, fauna",
            "bicycle, bike, wheel, cycle",
            "lake",
            "dishwasher, dish washer, dishwashing machine",
            "screen, silver screen, projection screen",
            "blanket, cover",
            "sculpture",
            "hood, exhaust hood",
            "sconce",
            "vase",
            "traffic light, traffic signal, stoplight",
            "tray",
            "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
            "fan",
            "pier, wharf, wharfage, dock",
            "crt screen",
            "plate",
            "monitor, monitoring device",
            "bulletin board, notice board",
            "shower",
            "radiator",
            "glass, drinking glass",
            "clock",
            "flag"], dtype=np.string)
      return obj[i]

def label_color(i):
      color = np.array([[120, 120, 120],
            [180, 120, 120],
            [  6, 230, 230],
            [ 80,  50,  50],
            [  4, 200,   3],
            [120, 120,  80],
            [140, 140, 140],
            [204,   5, 255],
            [230, 230, 230],
            [  4, 250,   7],
            [224,   5, 255],
            [235, 255,   7],
            [150,   5,  61],
            [120, 120,  70],
            [  8, 255,  51],
            [255,   6,  82],
            [143, 255, 140],
            [204, 255,   4],
            [255,  51,   7],
            [204,  70,   3],
            [  0, 102, 200],
            [ 61, 230, 250],
            [255,   6,  51],
            [ 11, 102, 255],
            [255,   7,  71],
            [255,   9, 224],
            [  9,   7, 230],
            [220, 220, 220],
            [255,   9,  92],
            [112,   9, 255],
            [  8, 255, 214],
            [  7, 255, 224],
            [255, 184,   6],
            [ 10, 255,  71],
            [255,  41,  10],
            [  7, 255, 255],
            [224, 255,   8],
            [102,   8, 255],
            [255,  61,   6],
            [255, 194,   7],
            [255, 122,   8],
            [  0, 255,  20],
            [255,   8,  41],
            [255,   5, 153],
            [  6,  51, 255],
            [235,  12, 255],
            [160, 150,  20],
            [  0, 163, 255],
            [140, 140, 140],
            [250,  10,  15],
            [ 20, 255,   0],
            [ 31, 255,   0],
            [255,  31,   0],
            [255, 224,   0],
            [153, 255,   0],
            [  0,   0, 255],
            [255,  71,   0],
            [  0, 235, 255],
            [  0, 173, 255],
            [ 31,   0, 255],
            [ 11, 200, 200],
            [255,  82,   0],
            [  0, 255, 245],
            [  0,  61, 255],
            [  0, 255, 112],
            [  0, 255, 133],
            [255,   0,   0],
            [255, 163,   0],
            [255, 102,   0],
            [194, 255,   0],
            [  0, 143, 255],
            [ 51, 255,   0],
            [  0,  82, 255],
            [  0, 255,  41],
            [  0, 255, 173],
            [ 10,   0, 255],
            [173, 255,   0],
            [  0, 255, 153],
            [255,  92,   0],
            [255,   0, 255],
            [255,   0, 245],
            [255,   0, 102],
            [255, 173,   0],
            [255,   0,  20],
            [255, 184, 184],
            [  0,  31, 255],
            [  0, 255,  61],
            [  0,  71, 255],
            [255,   0, 204],
            [  0, 255, 194],
            [  0, 255,  82],
            [  0,  10, 255],
            [  0, 112, 255],
            [ 51,   0, 255],
            [  0, 194, 255],
            [  0, 122, 255],
            [  0, 255, 163],
            [255, 153,   0],
            [  0, 255,  10],
            [255, 112,   0],
            [143, 255,   0],
            [ 82,   0, 255],
            [163, 255,   0],
            [255, 235,   0],
            [  8, 184, 170],
            [133,   0, 255],
            [  0, 255,  92],
            [184,   0, 255],
            [255,   0,  31],
            [  0, 184, 255],
            [  0, 214, 255],
            [255,   0, 112],
            [ 92, 255,   0],
            [  0, 224, 255],
            [112, 224, 255],
            [ 70, 184, 160],
            [163,   0, 255],
            [153,   0, 255],
            [ 71, 255,   0],
            [255,   0, 163],
            [255, 204,   0],
            [255,   0, 143],
            [  0, 255, 235],
            [133, 255,   0],
            [255,   0, 235],
            [245,   0, 255],
            [255,   0, 122],
            [255, 245,   0],
            [ 10, 190, 212],
            [214, 255,   0],
            [  0, 204, 255],
            [ 20,   0, 255],
            [255, 255,   0],
            [  0, 153, 255],
            [  0,  41, 255],
            [  0, 255, 204],
            [ 41,   0, 255],
            [ 41, 255,   0],
            [173,   0, 255],
            [  0, 245, 255],
            [ 71,   0, 255],
            [122,   0, 255],
            [  0, 255, 184],
            [  0,  92, 255],
            [184, 255,   0],
            [  0, 133, 255],
            [255, 214,   0],
            [ 25, 194, 194],
            [102, 255,   0],
            [ 92,   0, 255]], dtype=np.uint8)
      return color[i]

def direction(x, y):
      #輸入二維向量，計算該向量的絕對角度（0-360）
      if x==0 and y==0:
            direction = 0
      elif x==0:
            if y > 0:
                  direction = 90
            else:
                  direction = 270
      elif y==0:
            if x > 0:
                  direction = 360
            else:
                  direction = 180
      else:
            direction = math.atan2(y, x) * 180 / math.pi
            if direction<0:
                  direction += 360 

      return direction

def trans(value):
      if value==1:
            return 0
      elif value==0.5:
            return 1
      elif value==-0.5:
            return 2
      else:
            return 3

def trans_inv(value):
      if value==0:
            return 1
      elif value==1:
            return 0.5
      elif value==2:
            return -0.5
      else:
            return -0.1

class SceneModule():
      def __init__(self, rectangle_size, filename, seg_scale, scene_object_types):
            #rectangle_size: 框出長方形的長短邊長，橫向與直向
            #filename: 場景分割的檔案名稱包含副檔名
            #seg_scale: 場景分割圖的縮小倍率
            #scene_object_types: 場景中的物件類別數

            self.rectangle_size = rectangle_size
            self.seg_scale = seg_scale
            self.scene_object_types = scene_object_types

            seg_img = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.seg_height, self.seg_width = seg_img.shape[:2]
            
            # print(self.seg_width, self.seg_height)
            self.resize_width = int(self.seg_width/self.seg_scale)
            self.resize_height = int(self.seg_height/self.seg_scale)

            #確認原圖
            # cv2.imshow('My Image', seg_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            #將場景分割圖依倍率縮小
            seg_img = cv2.resize(seg_img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST)
            self.seg_img = seg_img
            
            #確認縮小圖            
            # cv2.imshow('My Image', seg_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            
            #HRNet所使用的色表，順序為RGB，而openCV的順序為BGR

            seg_array = np.zeros((self.resize_height, self.resize_width), dtype=int)
            # seg_list = np.zeros(150)
            
            for x in range(self.resize_width):
                  for y in range(self.resize_height):
                        for c in range(150):
                              if seg_img[y, x, 0] == label_color(c)[2] and seg_img[y, x, 1] == label_color(c)[1] and seg_img[y, x, 2] == label_color(c)[0]:
                                    if c==3 or c==6 or c==9 or c==11 or c==13: #4, 7, 10, 12, 14: 可行走
                                          seg_array[y, x] = 1
                                    elif c==12 or c==20 or c==116: #13, 21, 117: 交通物件
                                          seg_array[y, x] = 0.5
                                    elif c==0 or c==1 or c==4 or c==17 or c==32 or c==43 or c==87 or c==93:
                                          #1, 2, 5, 18, 33, 44, 88, 94: 障礙物
                                          seg_array[y, x] = -0.5
                                    else: #其他
                                          seg_array[y, x] = -0.1
                                    break
            
            self.seg_array = seg_array
            tmp = np.zeros((self.resize_height, self.resize_width, 3), np.uint8)

            for x in range(self.resize_width):
                  for y in range(self.resize_height):
                        if seg_array[y, x]==1:
                              tmp[y, x, 0] = 255
                        elif seg_array[y, x]==0.5:
                              tmp[y, x, 1] = 255
                        elif seg_array[y, x]==-0.5:
                              tmp[y, x, 2] = 255
                        else:
                              tmp[y, x, 0] = 255
                              tmp[y, x, 1] = 255
                              tmp[y, x, 2] = 255
                              
            cv2.imwrite(filename+'_converted.png', tmp)
                  
      def __call__(self, traj):
            return self.forward(traj)
      
      def forward(self, traj):
            #traj: 所有軌跡資料，為[步數, 封包軌跡數, 座標維度]的tensor，預設為[16, 封包軌跡數, 2]

            scene_tensor = torch.zeros([traj.size()[0], traj.size()[1], self.scene_object_types*3])
            # scene_tensor = torch.zeros([traj.size()[0], traj.size()[1], self.scene_object_types*2])
            # scene_tensor = torch.zeros([traj.size()[0], traj.size()[1], 2])
            #scene_tensor: 本模組的回傳值，包含每個位置周遭的各類別比例、重心距離與重心方向。
            #修改中

            for step in range(traj.size()[0]):
                  for ppl in range(traj.size()[1]):
                        #先取出軌跡中的座標
                        obj_x = int((traj[step, ppl, 0] * self.seg_width / self.seg_scale).item())
                        obj_y = int((traj[step, ppl, 1] * self.seg_height / self.seg_scale).item())

                        #考量到場景視角，車輛越遠面積越小，則該考慮的場景區域也該跟著變小
                        #尚待驗證
                        # if obj_y>5:
                        #       distance_scale = self.resize_height/obj_y
                        # else:
                        #       distance_scale = self.resize_height/5
                        distance_scale = 1
                        distance_width = self.rectangle_size[0]/distance_scale
                        distance_height = self.rectangle_size[1]/distance_scale

                        #利用座標與長方形邊長找出邊欲框出之邊界點
                        xStart = 0 if obj_x-distance_width/2<0 else math.ceil(obj_x-distance_width/2)
                        xEnd = self.resize_width-1 if obj_x+distance_width/2>=self.resize_width else math.ceil(obj_x+distance_width/2)
                        yStart = 0 if obj_y-distance_height/2<0 else math.ceil(obj_y-distance_height/2)
                        yEnd = self.resize_height-1 if obj_y+distance_height/2>=self.resize_height else math.ceil(obj_y+distance_height/2)
                        
                        #計算subtensor的中心點位置
                        xCenter = obj_x - xStart
                        yCenter = obj_y - yStart

                        #圖片驗證
                        # tmp_img = self.seg_img.copy()
                        # tmp_img = cv2.rectangle(tmp_img, (xStart, yStart), (xEnd, yEnd), (0, 0, 255), 1)
                        # cv2.imshow('My Image', tmp_img)
                        # cv2.waitKey(100)
                        # del tmp_img

                        #計算長方形中心點與最角落的最大距離
                        max_length = math.sqrt(distance_width*distance_width+distance_height*distance_height)

                        #計算長方形面積
                        area = (xEnd-xStart+1) * (yEnd-yStart+1) / 5 / 5

                        #以邊界點框出長方形
                        subseg = self.seg_array[yStart:yEnd, xStart:xEnd]

                        #紀錄長方形範圍中所有類別的位置加總與總個數
                        #total_position: 類別位置加總
                        #total_count: 類別數量加總
                        total_position = np.zeros((self.scene_object_types, 2))
                        total_count = np.zeros(self.scene_object_types)

                        for tmp_x in range(0, xEnd-xStart, 5):
                              for tmp_y in range(0, yEnd-yStart, 5):
                                    total_count[trans(subseg[tmp_y, tmp_x])] += 1
                                    total_position[trans(subseg[tmp_y, tmp_x]), 0] += (tmp_x - xCenter) * subseg[tmp_y, tmp_x]
                                    total_position[trans(subseg[tmp_y, tmp_x]), 1] += (tmp_y - yCenter) * subseg[tmp_y, tmp_x]

                        #紀錄每個類別的比例、重心距離與重心方向
                        for i in range(self.scene_object_types):
                              #類別比例
                              if area==0:
                                    scene_tensor[step, ppl, i*3] = 0
                              else:
                                    scene_tensor[step, ppl, i*3] = total_count[i] / area
                              #相對位置
                              if total_count[i]>0:
                                    position_x = total_position[i, 0] / total_count[i]
                                    position_y = total_position[i, 1] / total_count[i]
                              else:
                                    position_x = 0
                                    position_y = 0
                              #類別角度
                              #scene_tensor[step, ppl, i*3+1] = direction(position_x, position_y)/360
                              distance = math.sqrt(position_x*position_x + position_y*position_y)
                              #類別距離
                              if distance>0:
                                    scene_tensor[step, ppl, i*3+1] = position_x/distance
                                    scene_tensor[step, ppl, i*3+2] = position_y/distance
                        
                        '''# vector_x = 0
                        # vector_y = 0

                        #改為紀錄每個類別的重心與物件的相對位置，並乘上類別加成來作為方向參考依據的向量
                        for i in range(self.scene_object_types):
                              if area!=0:
                                    scale = total_count[i]/area
                                    #只在該類別總數不為零時紀錄
                                    if total_count[i]>0:
                                          #向量的x
                                          # vector_x += (total_position[i, 0] / total_count[i]) * trans_inv(i) * scale
                                          scene_tensor[step, ppl, 2*i] = (total_position[i, 0] / total_count[i]) * trans_inv(i) * scale
                                          #向量的y
                                          # vector_y += (total_position[i, 1] / total_count[i]) * trans_inv(i) * scale
                                          scene_tensor[step, ppl, 2*i+1]= (total_position[i, 1] / total_count[i]) * trans_inv(i) * scale
                        
                        # if vector_x!=0 or vector_y!=0:
                        #       scene_tensor[step, ppl, 0] = vector_x / math.sqrt(vector_x * vector_x + vector_y * vector_y)
                        #       scene_tensor[step, ppl, 1] = vector_y / math.sqrt(vector_x * vector_x + vector_y * vector_y)'''
                        del subseg
            return scene_tensor.cuda()
