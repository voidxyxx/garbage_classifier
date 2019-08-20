### 环境需求

	华为云平台moxing框架
	python 3.6以上
	pytorch-1.0.0
	Pillow
	numpy
	pandas

### 数据准备
1. 上传数据
	将训练数据上传至obs桶中，路径格式：s3://桶名/dataset/all/

2. 数据划分
	编辑./script/data_division.py，将以下代码：
		if __name__ == "__main__":
			data_divide(data_dir, train_dir, test_dir)#改成你系统里的对应目录
	三个路径改为对应路径，并在obs中建立对应文件夹
	data_dir = s3://桶名/dataset/all/
	train_dir = s3://桶名/dataset/trainset/
	test_dir = s3://桶名/dataset/testset/
	运行./script/data_division.py，将数据集随机划分为训练集（70%）和测试集（30%）

### 训练配置
1. 配置训练文件夹
	在obs桶中建立代码文件夹，路径格式：s3://桶名/code/
	将代码上传至对应文件夹中，注意，根据版本号进行划分
	如V0_0_3代码，代码路径为s3://桶名/code/V0_0_3/
	在obs桶中建立输出文件夹，路径格式：s3://桶名/training_output/
	注意，根据版本号进行划分
	如V0_0_3代码，代码路径为s3://桶名/training_output/V0_0_3/

2. 超参数配置
	根据训练表（./hyperpara.xlsx)配置模型超参数
	各超参数的修改方法如下：
	net-在train.py中，修改第69行
			net = resnet50()
	epochs-在train.py中，修改第117行
			train(epochs=5,
	init_lr-在train.py中，修改第118行
			init_lr=0.001,
	lr_coefficient-在train.py中，修改第119行
			lr_coefficient=5,
	weight_decay-在train.py中，修改第120行
			weight_decay = 1e-8,
	batch_size-在train.py中，修改第122行
			batch_size=64,
	transform-若transform为空，则保证在train.py中，修改第55行为
			trainset = DataClassify(train_dir, transforms=transform)
		若transform为std,则保证在train.py中，修改第55行为
			trainset = DataClassify(train_dir, transforms=transform_std)


3. 路径参数配置
	根据第2,3部的obs路径信息，修改模型对应路径
	需要修改的路径包括：
		train.py中，123-125行

### 训练

### 训练完成
1. 训练结果
	将华为云模型管理全部内容截图保存在对应版本号文件夹内
	将输出文件夹s3://桶名/training_output/内全部内容下载到对应版本号文件夹内
	调用你自己最初写的draw_and_save.py，画图并保存在对应版本号文件夹内

### 分析和调参

### 下一次迭代