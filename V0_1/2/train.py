import torch as t
from torch.nn import functional as F
from torch.autograd import Variable

from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from decayed_lr import dloss
from dataloader import *
from data_enhance import enhance_transforms, transform_standard
from evaluate import evaluate
from draw_save import curve_draw

def train_once(data_loader_train, net, optimizer, cost, device="cpu"):
	run_loss = 0.0
	run_correct = 0.0

	for data in iter(data_loader_train):
		X_train, X_label = data
		label = []
		for la in X_label:
			label.append(la)
		label = ListToTensor(label)

		X_train = Variable(X_train)
		X_label = Variable(label)
		X_train = X_train.to(device)
		X_label = X_label.to(device)
		optimizer.zero_grad()
		outputs = net(X_train)

		_, pred = t.max(F.softmax(outputs, dim=1).data, 1)
		loss = cost(outputs, X_label)
		run_loss += loss.data
		run_loss = run_loss.item()

		loss.backward()
		optimizer.step()
		run_correct += (pred == X_label.data).sum()
		corr = (1.*run_correct).item()
	return run_loss, corr


def train(epochs=120,
		init_lr=0.001,
		lr_coefficient=5,
		weight_decay = 1e-8,
		model_num=1,
		batch_size=64,
		train_dir='s3://classifier-gar/train_try/',
		test_dir='s3://classifier-gar/test_try/',
		log_dir='s3://classifier-gar/log/',
		version = 'V0_0_0'):

	#loading_data
	print("data loading...\n")
	transform = enhance_transforms()
	transform_std = transform_standard()
	trainset = DataClassify(train_dir, transforms=transform)
	testset = DataClassify(test_dir, transforms=transform_std)
	total_train = len(trainset)
	total_test = len(testset)
	data_loader_train = t.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
	data_loader_test = t.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
	print("data loading complete\n")

	##################################
	#TO DO
	##################################
	if model_num==0:
		exit(0)
	else:
		net = resnet50()
	##################################

	##################
	#cuda
	##################

	device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
	net.to(device)


	cost = t.nn.CrossEntropyLoss()
	train_loss_list = []
	train_accurate_list = []
	test_loss_list = []
	test_accurate_list = []
	
	for epoch in range(epochs):
		print("epoch " + str(epoch+1) + " start training...\n")

		net.train()

		learning_rate = dloss(train_loss_list, init_lr, lr_coefficient, init_lr)
		optimizer = t.optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)

		run_loss, corr = train_once(data_loader_train,net, optimizer, cost, device)
		train_loss_list.append(run_loss/total_train)
		train_accurate_list.append(corr/total_train)

		print('epoch %d, training loss %.6f, training accuracy %.4f ------\n' %(epoch+1, run_loss/total_train, corr/total_train))
		print("epoch " + str(epoch+1) + " finish training\n")
		print("-----------------------------------------------\n")
		

		print("epoch " + str(epoch+1) + " start testing...\n")
		
		net.eval()
		test_corr = evaluate(net, data_loader_test, device)
		test_accurate_list.append(test_corr/total_test)
		print('epoch %d, testing accuracy %.4f ------\n' %(epoch+1, test_corr/total_test))
		print("epoch " + str(epoch+1) + " finish testing\n")
		print("-----------------------------------------------\n")

	#torch.save(net.module, net_name)

	#torch.save(net.state_dict(), net_name_para)

	curve_draw(train_loss_list, train_accurate_list, test_accurate_list, log_dir, version)

	print("mission complete")


if __name__ == "__main__":
	train(epochs=30,
		init_lr=0.01,
		lr_coefficient=5,
		weight_decay = 1e-8,
		model_num=1,
		batch_size=64,
		train_dir='s3://obs-garbageclassification/obs-dataset/all/trainset/',
		test_dir='s3://obs-garbageclassification/obs-dataset/all/testset/',
		log_dir='s3://obs-garbageclassification/training_log/V0_1_2/',
		version = 'V0_1_2')