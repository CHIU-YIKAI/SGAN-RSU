import torch
import numpy as np

class TrajFreqMap():
    def __init__(self, trajMap, coord_scale=100):
        self.trajMap = trajMap
        self.coord_scale = coord_scale

    def __call__(self, traj, leng):
        return self.forward(traj,leng)

    def forward(self, traj, leng):
        # print(traj.size())
        map_w, map_h = self.trajMap.size()[0], self.trajMap.size()[1]
        if traj.dim()==3:
            Map_Info = torch.zeros([traj.size()[0], traj.size()[1], 9], dtype=torch.float32)
            for step in range(traj.size()[0]):
                for ppl in range(traj.size()[1]):
                    x,y = int((traj[step, ppl, 0]*self.coord_scale).item()), int((traj[step, ppl, 1]*self.coord_scale).item())
                    if x<0 or y<0 or x>map_w or y>map_h:
                        Map_Info[step, ppl, :] = torch.rand(9)
                    else: 
                        x += 3 * leng
                        y += 3 * leng
                        for i in range(3):
                            for j in range(3):
                                xStart = x - (3 - 2 * i) * leng
                                xEnd = x - (1 - 2 * i) * leng
                                yStart = y - (3 - 2 * j) * leng
                                yEnd = y - (1 - 2 * j) * leng
                                subtensor = self.trajMap[xStart:xEnd, yStart:yEnd]
                                Map_Info[step, ppl, i+3*j] = subtensor.mean()

        elif traj.dim()==2:
            Map_Info = torch.zeros([traj.size()[0],9])
            for step in range(traj.size()[0]):
                x,y = int((traj[step,0]*self.coord_scale).item()), int((traj[step,1]*self.coord_scale).item())
                if x<0 or y<0 or x>map_w or y>map_h:
                    Map_Info[step,:] = torch.rand(9)
                else: 
                    x += 3 * leng
                    y += 3 * leng
                    for i in range(3):
                        for j in range(3):
                            xStart = x - (3 - 2 * i) * leng
                            xEnd = x - (1 - 2 * i) * leng
                            yStart = y - (3 - 2 * j) * leng
                            yEnd = y - (1 - 2 * j) * leng
                            subtensor = self.trajMap[xStart:xEnd, yStart:yEnd]
                            Map_Info[step, i+3*j] = subtensor.mean()
                    
        return Map_Info.cuda()


        