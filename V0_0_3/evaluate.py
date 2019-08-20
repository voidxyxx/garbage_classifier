import torch as t
from torch.autograd import Variable

def evaluate(model, data_loader_test):
	model.eval()
	correct = 0.0
	run_correct = 0.0
	for data in iter(data_loader_test):
		X_test, X_label = data
		label = []
		for la in X_label:
			label.append(la)
		label = ListToTensor(label)
		X_test = Variable(X_test)
		X_label = Variable(label)
		with t.no_grad():
			outputs = model(X_test)
			_, pred = t.max(F.softmax(outputs, dim=1).data, 1)
			run_correct += (pred == X_label.data).sum()
			corr = (1.*run_correct).item()
	return corr