#coefficient = 5#超参数，学习率衰减系数，与模型有关，需要调参

def dloss(loss_list, lr=0.001, coefficient=5, init_lr=0.001):
	if len(loss_list) <=2:
		return init_lr#超参数,初始学习率，与训练样本集有关，与模型关系较小，确定训练集后即可固定
	if (loss_list[-2] - loss_list[-1])/loss_list[-2] > 0.02:
		return init_lr
	return lr/coefficient
