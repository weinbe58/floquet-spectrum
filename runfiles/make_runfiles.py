import numpy as np
from itertools import product






L_list = [64]
Delta_list = [1]
Omega_list = [1,1.5,2,4,8]
D_list = [4]

dV = 0.05
gamma = 0.05

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 50
	tr = int(10//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_1.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")




L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.05
gamma = 0.05

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 50
	tr = int(10//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_2.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")



L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma = 0.05

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 50
	tr = int(10//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

# with open("run_3.in","w") as IO:
# 	for job in jobs:
# 		IO.write(job+"\n")



L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma = 0.2

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

# jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 10
	tr = int(5//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_4.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")




L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma = 0.2

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 10
	tr = int(10//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_5.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")


L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma = 0.2

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 50
	tr = int(10//T + 1)
	tp = 5*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_6.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")



L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma = 0.2

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D in product(L_list,Delta_list,Omega_list,D_list):
	T = 2*np.pi/Omega
	
	tw = 0
	tr = int(10//T + 1)
	tp = 10*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_7.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")




L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma_list = [0.2,0.1,0.05]

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D,gamma in product(L_list,Delta_list,Omega_list,D_list,gamma_list):
	T = 2*np.pi/Omega
	
	tw = 0
	tr = int(1//T + 1)
	tp = 10*tr


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_8.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")



L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma_list = [0.05]

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D,gamma in product(L_list,Delta_list,Omega_list,D_list,gamma_list):
	T = 2*np.pi/Omega
	
	tw = int(np.ceil(5/T))
	tr = int(np.ceil(1/T))
	tp = int(np.ceil(10/T))


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_9.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")




L_list = [256]
Delta_list = [1,0.5]
Omega_list = [1,1.5,2,4,8]
D_list = [1]

dV = 0.01
gamma_list = [0.05]

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D,gamma in product(L_list,Delta_list,Omega_list,D_list,gamma_list):
	T = 2*np.pi/Omega
	
	tw = int(np.ceil(5/T))
	tr = int(np.ceil(1/T))
	tp = int(np.ceil(10/T))


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_9.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")



L_list = [256]
Delta_list = [1,0.5]
Omega_list = [4]
D_list = [1]
tr_list = [1,2,5]

dV = 0.01
gamma_list = [0.05,0.1]

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D,gamma,tr in product(L_list,Delta_list,Omega_list,D_list,gamma_list,tr_list):
	T = 2*np.pi/Omega
	
	tw = 0
	tr = int(np.ceil(tr/T))
	tp = int(np.ceil(15/T))


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)

	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_10.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")






L_list = [256]
Delta_list = [5]
Omega_list = [10]
D_list = [1]
tr_list = [1]

dV = 0.01
gamma_list = [0.05]

keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]
pars_fmt = "{{"+(",".join("'{key:}':{{{key:}:}}".format(key=k) for k in keys))+"}}"

jobs = []
for L,Delta,Omega,D,gamma,tr in product(L_list,Delta_list,Omega_list,D_list,gamma_list,tr_list):
	T = 2*np.pi/Omega
	
	tw = 0
	tr = int(np.ceil(tr/T))
	tp = int(np.ceil(15/T))


	BW = np.sqrt(Delta**2+2**2)
	V_max = np.ceil(BW+2*(Omega+2*BW))
	V_list = np.arange(-V_max,V_max+dV/10,dV)
	print(V_max)
	for V in V_list:
		pars = pars_fmt.format(L=L,Delta=Delta,D=D,Omega=Omega,V=np.around(V,10),tp=tp,tr=tr,tw=tw,gamma=gamma)
		jobs.append(pars)

with open("run_11.in","w") as IO:
	for job in jobs:
		IO.write(job+"\n")