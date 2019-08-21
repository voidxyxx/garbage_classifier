def dloss(loss_list, lr=0.001, coefficient=5, init_lr=0.001):
	if len(loss_list) <=2:
		return init_lr
	if (loss_list[-2] - loss_list[-1])/loss_list[-2] > 0.02:
		return init_lr
	return lr/coefficient
