# Rashmi Chaudhary

#//////////////////////////////////结合PSO///////////////////////

import random
import numpy
import pdb

from PIL import Image
import cv2
import array
import logging

# from deap import algorithms
# from deap import base
# from deap import benchmarks
# from deap import creator
# from deap import tools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centroids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


#
#
#    Fuzzy C-Means Implementation
#
#
class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=2, max_iterations=3, min_distance=5.0, size=200, m=2, epsilon=.001):
        self.k = k  # initialize k clusters
        self.max_iterations = max_iterations  # intialize max_iterations
        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = 0 #size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = epsilon # .001
        self.max_diff = 0.0
        self.image = 0
        # image_arr = numpy.array(self.image)
        # self.s = image_arr.size // image_arr.shape[2]


    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)   #改变图片大小
        image_arr=numpy.array(image)#将图像转换为数组的形式

        #计算一个通道的像素点数
        # print(image_arr.shape)
        if len(image_arr.shape)<3:#判断图片数据的通道数
            self.s=image_arr.size
        else:
            self.s=image_arr.size//image_arr.shape[2]  #取出其中一个通道的像素点，//计算结果是int类型的
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)#一个通道的所有像素点排成一排
            # self.beta = self.calculate_beta(self.image)

        print ("********************************************************************")
        for i in range(self.s):
            print (self.pixels[i])

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))#初始化uij
        randomPixels = random.sample(list(self.pixels), self.k)#从中随机选取k个像素点
        print("INTIALIZE RANDOM PIXELS AS CENTROIDS")
        print (randomPixels)
        #    print"================================================================================"
        for idx in range(self.k): #聚类中心的个数
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]#初始化ci
                # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print ("________", self.clusters[0].pixels[0])
        iterations = 0

        self.oldClusters = [cluster.centroid for cluster in self.clusters]
        print ("HELLO I AM ITERATIONS:", iterations)
        self.calculate_centre_vector()#第一次的更新uij和ci
        self.update_degree_of_membership()
        iterations += 1
        self.J_min=self.calculate_J(self.degree_of_membership,self.clusters)


        # shouldExit(iterations) checks to see if the exit requirements have been met.
            # - max iterations has been reached OR the centers have converged.
        while self.shouldExit(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print ("HELLO I AM ITERATIONS:", iterations)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            iterations += 1

        for cluster in self.clusters:#经过上述循环，满足要求之后，输出最后的聚类中中心ci
            print (cluster.centroid)
        return [cluster.centroid for cluster in self.clusters]

    def selectSingleSolution(self):
        self.max_iterations = 10

    def shouldExit(self, iterations):
        
        '''#/////////////////////////根据迭代次数和uij作为循环结束标志///////////////
        if iterations >= self.max_iterations or self.max_diff < self.epsilon:#迭代次数到达上限或者uij变化小于阈值就退出循环
            return True
        print("delta max_diff:",self.max_diff)#输出此次迭代输出的uij差值
        return False
        '''
        self.clusters=self.calculate_centre_vector()
        self.degree_of_membership=self.update_degree_of_membership()

        self.J =self.calculate_J(self.degree_of_membership,self.clusters)

        # if self.J<self.J_min:
        #     self.J_min=self.J

        self.diff=abs(self.J-self.J_min)
        if self.diff<self.epsilon or iterations >= self.max_iterations:
            return True
        print("delta diff:", self.diff)

        return False
        # if (self.max_diff > self.epsilon):
        #   return False
        # Perform normalization
        #self.normalization()
        # for i in self.s:


    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        t = []
        for i in range(self.s):
            t.append([])
            for j in range(self.k):
                t[i].append(pow(self.degree_of_membership[i][0][j], self.m))
        # print"\n\nCALC_CENTRE_VECTOR INVOKED:"

        for cluster in range(self.k):
            # print"*********************************************************************************"
            numerator = 0.0
            denominator = 0.0
            for i in range(self.s):
                # print "+++++++++", self.clusters[cluster].pixels[i], t[i][cluster], (t[i][cluster] * self.clusters[cluster].pixels[i])
                numerator += t[i][cluster] * self.clusters[cluster].pixels[i]
                denominator += (t[i][cluster])
                # print " ______ ", numerator/denominator
            self.clusters[cluster].centroid = (numerator / denominator)
        return self.clusters

    # Updates the degree of membership for all of the data points.
    def update_degree_of_membership(self,pixels,clusters,s,k):
        self.max_diff = 0.0
        degree_of_membership=[]
        for i in range(s):
            degree_of_membership.append(numpy.random.dirichlet(numpy.ones(k), size=1))#初始化uij,这边主要是为了开辟对应的空间存储uij

        for j in range(s):
            for idx in range(k):#上一层取一个样本，这个样本到所有中心的距离
                new_uij = self.get_new_value(pixels[j], clusters[idx].centroid,clusters)#计算新的uij
                if (j == 0):
                    print ("This is the Updatedegree centroid number:", idx, clusters[idx].centroid)#计算uij时候，先把所导进来的ci输出显示
                #////////////////这边的终止条件暂时没用用到//////////
                # diff = new_uij - self.degree_of_membership[j][0][idx]
                # if (diff > self.max_diff):
                #     self.max_diff = diff
                #//////////////////////////////////////////////////
                degree_of_membership[j][0][idx] = new_uij
        # return self.max_diff    #原代码
        return degree_of_membership

    def get_new_value(self, i, j,z):#计算新的uij
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in z:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)#val得p次方
            sum += val
        return (1.0 / sum)

    def normalization(self):
        max = 0.0
        highest_index = 0
        for i in range(self.s):
            # Find the index with highest probability
            for j in range(self.k):
                if (self.degree_of_membership[i][0][j] > max):
                    max = self.degree_of_membership[i][0][j]
                    highest_index = j
            # Normalize, set highest prob to 1 rest to zero
            for j in range(self.k):
                if (j != highest_index):
                    self.degree_of_membership[i][0][j] = 0
                else:
                    self.degree_of_membership[i][0][j] = 1

    def calculate_J(self,degree_of_membership,clusters,pixels,k,s):
        J=0
        for j in range(s):
            for i in range(k):
                J=numpy.add(pow(degree_of_membership[j][0][i],self.m)*pow(self.calcDistance(pixels[j],clusters[i].centroid),2),J)
        return J


    # Shows the image.
    def showImage(self):
        self.image.show()

    def showClustering(self,image,pixels,clusters):
        localPixels = [None] * len(image.getdata())
        for idx, pixel in enumerate(pixels):
            shortest = float('Inf')
            for cluster in clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster
            #print "cluster ", cluster, nearest.centroid
            localPixels[idx] = nearest.centroid#将聚类中心点附近的像素点都设置成于最近的ci相同的值

        w, h = image.size

        # if (len(numpy.array(image).shape)<3):
        #     localPixels = numpy.asarray(localPixels) \
        #         .astype('uint8') \
        #         .reshape((h, w))
        # else:
        #     localPixels = numpy.asarray(localPixels) \
        #         .astype('uint8') \
        #         .reshape((h, w, numpy.array(image).shape[2]))

        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w))
        colourMap = Image.fromarray(localPixels)
        #colourMap.show()
        cluster_img=numpy.array(colourMap)
        # plt.imsave("outputimage.png",colourMap)#原代码，但是在服务器上有问题
        # cv2.imwrite("outputimage_-w.png",numpy.array(colourMap))#这个可以在服务器上执行
        return cluster_img

    def showScatterPlot(self):
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        sum_of_overlapeed_pixels = 0

        for i in range(self.s):
            # Find the index with highest probability
            status = False
            for j in range(self.k):
                if self.degree_of_membership[i][0][j] >=0.5:
                    status = True

            if status == False:
                sum_of_overlapeed_pixels = sum_of_overlapeed_pixels+1

        print ("sum of overlapped pixels ", sum_of_overlapeed_pixels )


if __name__ == "__main__":
    # image = Image.open("4.png")#原代码，用来直接读取原始数据

    image=cv2.imread("he_1.png")
    cv2.imwrite("he_1b.png",image)#读写一次改变图片的通道数
    image = Image.open("he_1b.png")

    f = fcm()
    # print(image.shape)
    result = f.run(image)
    f.showScatterPlot()#原代码
    f.showClustering()
    # print f.I_index()
    # print f.JmFunction()
    # print f.XBindex()