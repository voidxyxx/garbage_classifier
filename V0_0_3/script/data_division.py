#为保证原有数据安全，此脚本产生划分后的数据备份
#样本具有保序性

import os
import shutil
import random

def data_divide(data_dir, train_dir, test_dir):
	for i in range(1000):#数据集最后一项的数字，不是数据集总数
		imgpath = data_dir + str(i) + ".jpg"
		txtpath = data_dir + str(i) + ".txt"
		if os.path.isfile(imgpath):
			if random.randint(0, 9) < 3:#30%概率数据选中为
				shutil.copy(imgpath, test_dir)
				shutil.copy(txtpath, test_dir)
				print("No." + str(i) + " has been divided into testset\n")
			else:
				shutil.copy(imgpath, train_dir)
				shutil.copy(txtpath, train_dir)
				print("No." + str(i) + " has been divided into trainset\n")

if __name__ == "__main__":
	data_divide(data_dir, train_dir, test_dir)#改成你系统里的对应目录
	#data_dir：原始数据
	#train_dir：训练集
	#test_dir：测试集