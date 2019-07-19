import random
import numpy as np
from FCM_Test2 import fcm
import cv2
from PIL import Image
from FCM_Test2 import Cluster
import matplotlib.pyplot as plt

class PSO_optimization(object):

	random.seed()

	### Initialization of positions ###
	def initPos(self,image, m,k):# m是种群的数量，k是聚类中心个数
		# image=np.array(image)
		# self.width = image.shape[1]
		# self.height= image.shape[0]
		# print("image size:",image.shape)

		self.pixels = image.getdata()
		self.Max_Pixel=max(self.pixels)[0]
		self.Min_Pixel=min(self.pixels)[0]
		self.pixels=np.array(self.pixels, dtype=np.uint8)

		# self.particles =np.zeros((m,k),dtype=np.int)
		self.particles=np.zeros((m, k), dtype=np.float)
		for i in range(m):
			for j in range(k):
				self.particles[i][j] =  random.uniform(self.Min_Pixel,self.Max_Pixel+random.randint(0,1)) # 在图中随机选择m个不同的数据点，即作为一个种群,randint能取到上下界
				while random.randint(self.Min_Pixel,self.Max_Pixel)+ random.uniform(0,1) in self.particles:#防止选择的是同一个种群
					self.particles[i][j] = random.uniform(self.Min_Pixel, self.Max_Pixel+random.randint(0,1))
				# list(self.particles[i]).append([x,y])
				# self.particles[i][j]=self.Position2PixelIndex(x,y)
		return self.particles

	### Initialization of velocity vector ###
	def initVel(self,m, V_max):
		return [[random.uniform(0,V_max) for j in range(k)] for i in range(m)]

	def PSO(self,iterations, m, k, image, W=-0.8, V_max = 3, c1=1.3, c2=1.3):#这边的m，用在FCM时候，m个ci作为一个个体
		self.f = fcm()
		# degree_of_membership=[]

		clusters = [[None for i in range(k)] for i in range(m)] #m行，k列，存储聚类中心

		positions = self.initPos(image,m,k)#初始化种群

		image_arr = np.array(image)  # 将图像转换为数组的形式
		self.s = image_arr.size // image_arr.shape[2]#计算政府图片的像素个数


		degree_of_membership = []
		for i in range(self.s):
			degree_of_membership.append(np.random.dirichlet(np.ones(k), size=1))
		#///////////计算uij

		for i in range(m):
			for idx in range(k):  # 聚类中心的个数
				clusters[i][idx] = Cluster()
				clusters[i][idx].centroid = positions[i][idx]
				# clusters[i][idx].centroid =pixels[positions[i][idx]]

		#/////////////////////////////PSO迭代计算/////////////////


		for i in range(m):
			degree_of_membership[i]=self.f.update_degree_of_membership(self.pixels,clusters[i],self.s,k)

		#///////////计算目标函数func的值
		func=[]
		for i in range(m):
			func.append(self.f.calculate_J(degree_of_membership[i],clusters[i],self.pixels,k,self.s))

		MinFunc_index=func.index(min(func)) #返回目标函数中最小的那个种群下标
		global_best=positions[MinFunc_index]
		p_best=positions
		# best = [(f.calculate_J(degree_of_membership[i],clusters[i],pixels,k,s), clusters[i] ) for i in m]
		#///////////更新positions////

		# best = [(func(image,elem[0],elem[1]),elem[0],elem[1]) for elem in positions]#记录所有种群个体的func计算的值
		# global_best = max(best, key=lambda x:x[0])#这边可能得改为min，计算的其中的最值作为全局最佳值
		vel = self.initVel(m, V_max)
		J_value = []
		for x in range(iterations):
			# p_best=p_best
			print("HELLO I AM ITERATIONS:", x)
			cur_positions=positions#先把当前位置保存下来
			positions = np.zeros((m, k), dtype=np.float)#[]
			for i in range(m):#迭代iterations次，选择最佳的个体作为种群的最佳状态，下面是对v和x的更新

				# 	k_pixels=[]
				vel[i] = [
					np.multiply(W,vel[i])+                        #vi=wvi+c1*rand()*(pbesti-xi)+c2*rand()*(gbesti-xi)
					np.multiply(np.multiply(c1,[random.uniform(0,1) for j in range(k)]),
					(np.subtract(p_best[i],cur_positions[i])))+
					np.multiply(np.multiply(c2,[random.uniform(0,1) for j in range(k)]),
					(np.subtract(global_best, cur_positions[i])))]
					#for i in range(m)]
				positions[i]=np.add(cur_positions[i],vel[i])  # xi=xi+vi
					# for j in range(k):
					# 	k_pixels.append(self.pixels[self.Position2PixelIndex(self.positions[i][j][0], self.positions[i][j][1])])
				#//////////////////判断位置，保证不越界
				for j in range(k):
					if positions[i][j]>self.Max_Pixel:#防止超出最大边缘
						positions[i][j]=self.Max_Pixel
					elif positions[i][j]<self.Min_Pixel:
						positions[i][j]=self.Min_Pixel

			print("---------------------Next is the p_best value--------------------")
			p_best=self.Positions2p_best(cur_positions,positions,m,k)#PSO更新完之后得到的positions得根据func在判断是否是p_best
			print("p_best:",p_best)
			print("---------------------Next is the global_best value--------------------")
			global_best=self.P_best2global_best(p_best,m,k)#上述m个种群运算完之后，根据p_best计算global_best
			print("global_best",global_best)

			J_value.append(self.Single_func(global_best, k))
			if self.Single_func(global_best,k)<1:
				break

				#////////////////////////////判断并重新更新p_best

				# best = [(func(image,elem[0], elem[1]), elem[0], elem[1]) for elem in positions]#重新计算种群中各个个体的数值
				# global_best = max(best+[global_best], key=lambda x: x[0])#重新选择最佳的个体
			vel = self.initVel(m, V_max)
		return global_best,J_value

	def Positions2p_best(self,cur_positions,positions,m,k):
		p_best=np.zeros((m, k), dtype=np.float)
		cur_func=self.Func_calculate(cur_positions,m,k)#计算相应的目标函数值
		update_func=self.Func_calculate(positions,m,k)
		d=np.array(cur_func)-np.array(update_func)

		for i in range(m):
			if d[i]<=0:
				p_best[i]=cur_positions[i]
			else:
				p_best[i]=positions[i]

		return p_best

	def P_best2global_best(self,p_best,m,k):
		p_bestFunc=self.Func_calculate(p_best,m,k)
		MinFunc_index=p_bestFunc.index(min(p_bestFunc))
		global_best=p_best[MinFunc_index]

		return global_best

	def Func_calculate(self,positions,m,k):
		degree_of_membership = []
		clusters = [[None for i in range(k)] for i in range(m)]

		for i in range(self.s):
			degree_of_membership.append(np.random.dirichlet(np.ones(k), size=1))

		for i in range(m):
			for idx in range(k):  # 聚类中心的个数
				clusters[i][idx] = Cluster()
				# print("Now index x,y:",positions[i][idx][0],positions[i][idx][1])
				clusters[i][idx].centroid = positions[i][idx]#根据坐标计算对应的像素值

		for i in range(m):
			degree_of_membership[i]=self.f.update_degree_of_membership(self.pixels,clusters[i],self.s,k)
		#///////////计算目标函数func的值
		func=[]
		for i in range(m):
			func.append(self.f.calculate_J(degree_of_membership[i],clusters[i],self.pixels,k,self.s))
		return func

	def Position2PixelIndex(self,x,y):
		Pixel_index=self.width*y+x
		return Pixel_index

	def Show_image(self,image,pixels,global_best):
		self.f.showClustering(image,pixels,global_best)

	def Single_func(self,global_best,k): #通过一个聚类中心，然后计算其对应的func值
		clusters = [None for i in range(k)]
		for idx in range(k):  # 聚类中心的个数
			clusters[idx] = Cluster()
			clusters[idx].centroid = global_best[idx]
		degree_of_membership = self.f.update_degree_of_membership(self.pixels, clusters, self.s, k)
		J_value=self.f.calculate_J(degree_of_membership,clusters,self.pixels,k,self.s)
		return  J_value
if __name__ == "__main__":
	# # image = Image.open("4.png")#原代码，用来直接读取原始数据
    #
	# image=cv2.imread("he_4.png")
	# cv2.imwrite("he_4b.png",image)#读写一次改变图片的通道数
	# image = Image.open("he_4b.png")
    #
	# f = fcm()
	# # print(image.shape)
	# result = f.run(image)
	# f.showScatterPlot()#原代码
	# f.showClustering()
	iterations=8
	m=3
	k=3
#//////////////////这边读写一遍是为了修改图片的位数，因为8为不能给 Image.open打开////////////
	image1 = cv2.imread("he_6.png")
	cv2.imwrite("he_6b.png", image1)
	image2 = Image.open("he_6b.png")
	p=PSO_optimization()
	pixels = np.array(image2.getdata(), dtype=np.uint8)
	global_best,J_value=p.PSO(iterations,m,k,image2)#返回的是最佳聚类中心的坐标位置，和最佳位置对应的func值
#///////////////画目标函数func与迭代次数关系图///////////////
	x = []#存储迭代次数，作为x轴
	for i in range(iterations):
		x.append(i)
	plt.plot(x,J_value)
	plt.xlabel("iterations")
	plt.ylabel("func_value")
	plt.title("func_iteration")
	plt.savefig("PSO2_func_iteration"+str(iterations)+"_"+str(m)+"_"+str(k)+".png")
#///////////////////////////////////////////////////////////
##////////////////////显示最后的聚类结果/////////////////////
	print("final global_best location:",global_best)
	# print("image2 shape:",np.array(image2).shape)
	#////////////////输出全局最佳聚类中心的像素值
	clusters = [None for i in range(k)]
	for i in range(k):
		clusters[i]=Cluster()
		clusters[i].centroid=global_best[i]
	print("final global_best location pixels:")
	for cluster in clusters:  # 经过上述循环，满足要求之后，输出最后的聚类中中心ci
		print(cluster.centroid)
	#保存聚类图片
	p.Show_image(image2,pixels,clusters)
