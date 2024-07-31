import os, sys, random
import numpy as np
import preprocessing

from keras.datasets import mnist, fashion_mnist, cifar10
from sklearn import datasets

# dataset name as the argument
if sys.argv[1].lower() == 'all':
	all_datasets = ['mnist', 'kmnist', 'fmnist', 'cifar2', 'kws', 'iris', 'battery', 'emg', 'sports']
else:
	all_datasets = [sys.argv[1].lower()]

# make directory to store the booleanized dataset(s)
if not os.path.exists(r"bool_datasets"):
	os.makedirs(r"bool_datasets")

for dataset in all_datasets:
	match dataset:
	
		# MNIST
		case "mnist":
			(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# KMNIST
		case "kmnist":
			X_train = np.load(r"kmnist/kmnist-train-imgs.npz")['arr_0']
			X_test = np.load(r"kmnist/kmnist-test-imgs.npz")['arr_0']
			Y_train = np.load(r"kmnist/kmnist-train-labels.npz")['arr_0']
			Y_test = np.load(r"kmnist/kmnist-test-labels.npz")['arr_0']
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# FMNIST
		case "fmnist":
			(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# IRIS
		case "iris":
			iris = datasets.load_iris()
			X = preprocessing.onehot_encoding(iris.data, 3)
			Y = iris.target
			random.seed(1)
			n = random.sample(range(len(Y)), len(Y))
			
			X_train = []
			Y_train = []
			for each in n[0:int(len(Y)*0.8)]:
				X_train.append(X[each])
				Y_train.append(Y[each])

			X_test = []
			Y_test = []
			for each in n[int(len(Y)*0.8):]:
				X_test.append(X[each])
				Y_test.append(Y[each])

			X_train=np.array(X_train)
			Y_train=np.array(Y_train)
			X_test=np.array(X_test)
			Y_test=np.array(Y_test)

		# Battery capacity estimation
		case "battery":
			X_train_raw = preprocessing.battery_dataset(r"battery/25deg_XTrain_SAMPLED_Pana.txt")
			Y_train = preprocessing.battery_dataset(r"battery/25deg_YTrain_Pana_TM_Class.txt")
			Y_train = Y_train.flatten()
			X_test_raw = preprocessing.battery_dataset(r"battery/25deg_XTest1_SAMPLED_Pana.txt")
			Y_test = preprocessing.battery_dataset(r"battery/25deg_YTest_Pana_TM_Class.txt")
			Y_test = Y_test.flatten()

			X_raw_features = np.concatenate((X_train_raw, X_test_raw), axis=0)
			X = preprocessing.thermo_encoding(X_raw_features, 40)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			Y_train=np.array(Y_train)
			X_test=np.array(X_test)
			Y_test=np.array(Y_test)

		# keyword spotting
		case "kws":
			[train_x, Y_train, test_x, Y_test] = preprocessing.kws_dataset(r"data/mini_speech_commands")
			X_raw_features = np.concatenate((train_x, test_x), axis=0)
			X = preprocessing.thermo_encoding(X_raw_features, 3)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)

		# CIFAR2
		case "cifar2":
			(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
			Y_train=Y_train.reshape(Y_train.shape[0])
			Y_test=Y_test.reshape(Y_test.shape[0])
			animals = np.array([2, 3, 4, 5, 6, 7])
			Y_train = np.where(np.isin(Y_train, animals), 1, 0)
			Y_test = np.where(np.isin(Y_test, animals), 1, 0)
			X_raw_features = np.concatenate((X_train_org, X_test_org), axis=0)
			X = preprocessing.CIFAR_HOG(X_raw_features)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)

		# EMG
		case "emg":
			train = np.loadtxt(r'TinyML/EMG/EMG_train.txt')
			test = np.loadtxt(r'TinyML/EMG/EMG_test.txt')
			X_train = train[:, :-1]
			Y_train = train[:, -1]
			X_test = test[:, :-1]
			Y_test = test[:, -1]

		# Sports
		case "sports":
			train = np.loadtxt(r'TinyML/Sports/Sports_train.txt')
			test = np.loadtxt(r'TinyML/Sports/Sports_test.txt')
			X_train = train[:, :-1]
			Y_train = train[:, -1]
			X_test = test[:, :-1]
			Y_test = test[:, -1]		
										
		case _:
			print("The given dataset %s is not recognized." %sys.argv[1])

	# store datasets in given directory
	if not os.path.exists(r"bool_datasets/"+dataset):
		os.makedirs(r"bool_datasets/"+dataset)
	np.save(r"bool_datasets/"+dataset+'/X_train.npy', X_train)
	np.save(r"bool_datasets/"+dataset+'/Y_train.npy', Y_train)
	np.save(r"bool_datasets/"+dataset+'/X_test.npy', X_test)
	np.save(r"bool_datasets/"+dataset+'/Y_test.npy', Y_test)