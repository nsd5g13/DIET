
W = 32

# W datapoints for each inference
datapoint_file = 'redress/X_test.txt'
f = open(datapoint_file)
datapoints = f.readlines()
number_of_features = len(datapoints[0])-1	# number of features, f
W_datapoints = []
for i in range(int(len(datapoints)/W)):
	W_dps = ''
	for j in range(W):
		W_dps = W_dps + datapoints[i*W+j].replace('\n', '')
	W_datapoints.append(W_dps)
f.close()

# number of includes for each class
includes_classes_file = 'redress/no_includes.txt'
f = open(includes_classes_file)
includes = f.readlines()
number_of_classes = len(includes)	# number of classes, M
includes_per_class = []
for each in includes:
	includes_per_class.append(int(each.replace('\n', '')))
f.close()

# redress include codes
encode_file = 'redress/encoded_include.txt'
f = open(encode_file)
codes = f.readlines()
IncEncs = []
cnt = 0
for i in range(number_of_classes):
	IncEncs.append([])
	for code in codes[cnt : cnt+includes_per_class[i]]:
		IncEncs[i].append(code.replace('\n', ''))
	cnt = cnt + includes_per_class[i]
f.close()

inferred_class = []
# W inferences simultaneously
for datapoints in W_datapoints:
	# initialization
	cur_class_sum = [0 for i in range(W)]
	classification = [0 for i in range(W)]

	# inference
	for i in range(number_of_classes):
		cl_output = [1 for j in range(W)]
		prev_cl = 1
		cl_polarity = 0

		for IncEnc in IncEncs[i]:
			curr_cl = IncEnc[1]

			# if clause bit flipped
			if curr_cl != prev_cl:
				for j in range(W):
					each_cl_output = cl_output[j]
					cur_class_sum[j] = cur_class_sum[j] + (1-2*cl_polarity)*cl_output[j]
				cl_output = [1 for j in range(W)]
		
			cl_polarity = int(IncEnc[0])
			feature_offset = int(IncEnc[2:15], 2)
			literal_polarity = IncEnc[15]
			prev_cl = curr_cl

			# compute clause output
			if literal_polarity == '0':
				features_in_datapoints = [int(datapoints[number_of_features*j+feature_offset]) for j in range(W)]
			else:
				features_in_datapoints = [1-int(datapoints[number_of_features*j+feature_offset]) for j in range(W)]
			cl_output = [output*feature for output, feature in zip(cl_output, features_in_datapoints)]

		# argmax
		if i == 0:
			class_sum = cur_class_sum
		else:
			for j in range(W):
				if class_sum[j] < cur_class_sum[j]:
					class_sum[j] = cur_class_sum[j]
					classification[j] = i
		cur_class_sum = [0 for i in range(W)]
		
	inferred_class.extend(classification)
