import sys, os

no_classes = int(sys.argv[1])
no_clauses = int(sys.argv[2])
no_features = int(sys.argv[3])
vanillaORdiet = sys.argv[4]

if vanillaORdiet.lower() == 'vanilla':
	f = open(r"../VanillaTM_training/log/clause_expr.txt", "r")
elif vanillaORdiet.lower() == 'diet':
	f = open(r"../DIET_training/log/clause_expr.txt", "r")
else:
	print("Choose the clause expressions produced by either vanilla TM (vanilla) or DIET (diet)")
f_lines = f.readlines()

c1 = 0
c2 = 0
output_list = []
is_next_expr = False
verilog_content = ""
for each in f_lines:
	# Generage verilog logic expression for each clause
	if is_next_expr == True:
		is_next_expr = False
		if len(each.split())>0:
			verilog_content = verilog_content + ("assign c%d%d = " %(c1, c2)) + each[:-1] + ";\n"

	if len(each.split()) > 1:
		# Class no and Clause no
		if each.split()[0] == "Class":
			c1 = int(each.split()[1])
			is_next_expr = False
		elif each.split()[0] == "Clause":
			c2 = int(each.split()[1].replace("#", "").replace(":", ""))
			is_next_expr = True
	
	output_list.append("c%d%d" %(c1,c2))

# input ports
input_list = []
for i in range(no_features):
	input_list.append("x"+str(i))

# output ports
output_list = (list(set(output_list)))

# module definition
module_def = "module tm( " + ",".join(input_list) + "," + ",".join(output_list) + " );\n"

# port definition
port_def = ""
for each in input_list:
	port_def = port_def + "input " + str(each) + ";\n"
for each in output_list:
	port_def = port_def + "output " + str(each) + ";\n"

verilog_content = module_def + "\n" + port_def + "\n" + verilog_content + "\nendmodule"
if not os.path.exists(r"hdl"):
	os.makedirs(r"hdl")
verilog_file = open(r"hdl/tm.v", "w")
verilog_file.write(verilog_content)
verilog_file.close()
