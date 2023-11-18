import numpy as np
import scipy.sparse as sp
import scipy.integrate as inte
import os,h5py,sys
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor,MPIPoolExecutor,as_completed
from quspin.tools.evolution import evolve

def smooth_step(t,s):
	return (np.exp(-(s/t)**2) if t>0 else 0)


def _probe(k,V,Omega,Delta,gamma,D,tw,tr,tp):

	def eom(t,y,H0,G,Ek,K,A,Gamma):

		ek = -2*np.cos(0.5*K+A(t))

		psi = y.reshape((4,-1))

		return -1j*(np.dot(H0,psi)+np.dot(Gamma(t)*G,psi)+np.dot(Ek,ek*psi)).ravel()


	Sz = np.array([[1,0],[0,-1]])
	Sx = np.array([[0,1],[1,0]])
	Id = np.eye(2)

	Nk = k.size


	T = 2*np.pi/Omega
	tw = np.asarray(T*tw,dtype=np.float64)
	tr = np.asarray(T*tr,dtype=np.float64)
	tp = np.asarray(T*tp,dtype=np.float64)


	ek = -2*np.cos(k/2)


	H0 = 0.5*Delta*np.kron([[1,0],[0,0]],Sz)+V*np.kron([[0,0],[0,1]],Id)
	G = np.kron(Sx,Id)
	Ek = np.kron([[1,0],[0,0]],Sx)



	theta = np.arctan2(ek,0.5*Delta)



	psi = np.zeros((4,Nk),dtype=np.complex128)
	psi[0,...] = -np.sin(theta/2)
	psi[1,...] =  np.cos(theta/2)

	psi = psi.ravel()



	A = lambda t:D*np.sin(Omega*t)
	Gamma = lambda t:gamma*smooth_step(t,tr)

	args = (H0,G,Ek,k,A,Gamma)

	times = [tp]
	psi_t = inte.solve_ivp(eom,(-tw,tp),psi,args=args,t_eval=times,atol=1e-13,rtol=1e-13,method="DOP853")


	psi = psi_t.y.reshape((4,Nk))
	n = np.abs(psi)**2

	return n,()

def _Floquet_eigenvalues(k,Omega,Delta,D):

	def eom(t,y,H0,Ek,K,A):

		ek = -2*np.cos(0.5*K+A(t))

		psi = y.reshape((2,-1))

		return -1j*(H0.dot(psi) + ek * Ek.dot(psi)).ravel()


	Sz = np.array([[1,0],[0,-1]])
	Sx = np.array([[0,1],[1,0]])

	Nk = k.size


	T = 2*np.pi/Omega

	k = np.vstack((k,k)).T.ravel()

	H0 = 0.5*Delta*Sz
	Ek = Sx

	U0 = np.zeros((2,2*Nk),dtype=np.complex128)
	U0[0,0::2] = 1
	U0[1,1::2] = 1

	A = lambda t:D*np.sin(Omega*t)

	args = (H0,Ek,k,A)

	times = [T]

	# Ut = evolve(U0.ravel(),0,times,eom,f_params=args)
	sol = inte.solve_ivp(eom,(0,T),U0.ravel(),args=args,t_eval=times,atol=3e-14,rtol=3e-14,method="DOP853")
	Ut = sol.y

	Ut = Ut.reshape((2,-1))
	U = np.zeros((2,2,Nk),dtype=np.complex128)

	U[:,0,...] = Ut[:,0::2]
	U[:,1,...] = Ut[:,1::2]


	U = np.transpose(U,(2,0,1))

	q,v = np.linalg.eig(U)

	e = (np.log(q)/(-1j*T)).real

	e.sort(axis=1)
	print(e.shape)

	return e,()






def _Floquet_fourier(modes,k,Omega,Delta,D):

	def eom(t,y,H0,Ek,K,A):

		ek = -2*np.cos(0.5*K+A(t))

		psi = y.reshape((2,-1))

		return -1j*(H0.dot(psi) + ek * Ek.dot(psi)).ravel()


	Sz = np.array([[1,0],[0,-1]])
	Sx = np.array([[0,1],[1,0]])

	Nk = k.size


	T = 2*np.pi/Omega


	k = np.vstack((k,k)).T.ravel()

	H0 = 0.5*Delta*Sz
	Ek = Sx

	U0 = np.zeros((2,2),dtype=np.complex128)
	U0[0,0] = 1
	U0[1,1] = 1

	A = lambda t:D*np.sin(Omega*t)

	args = (H0,Ek,k,A)


	sol = inte.solve_ivp(eom,(0,T),U0.ravel(),args=args,dense_output=True,atol=3e-14,rtol=3e-14,method="DOP853")
	U = sol.sol(T).reshape(2,2)

	q,v = np.linalg.eig(U)

	e = (np.log(q)/(-1j*T)).real

	fourier = np.zeros(modes.shape+(4,))

	for i,m in enumerate(modes):

		f = lambda t:sol.sol(t)*np.exp(-1j*m*Omega*t)

		fourier[m] = inte.quad_vector(f,0,T)/T

	return e,fourier,()


def Floquet_probe(L,V,Omega,Delta,gamma,D,tw,tr,tp):
	k = np.arange(-L//2+1,L//2+1,1)*2*np.pi/L
	return _probe(k,V,Omega,Delta,gamma,D,tw,tr,tp)

def Floquet_eigenvalues(L,Omega,Delta,D):
	k = np.arange(-L//2+1,L//2+1,1)*2*np.pi/L
	return _Floquet_eigenvalues(k,Omega,Delta,D)

def main():
	comm = MPI.COMM_WORLD

	with MPICommExecutor(comm,root=0) as executor:
		if executor is not None:
			runfile = sys.argv[1]

			# first pass to get the unfinished jobs
			jobs = []
			with h5py.File("data.hdf5","a") as database:
				for line in open(runfile,"r"):
					pars = eval(line)
					V = pars["V"]

					par_no_V = dict(pars)
					del par_no_V["V"]


					grp = "Bandmodel_{}".format(str(par_no_V))
					if  grp not in database:
						database.create_group(grp)


					name = "V_{:}".format(V)
					keys = ["L","V","Omega","Delta","gamma","D","tw","tr","tp"]


					if name not in database[grp]:
						jobs.append((grp,name,tuple(pars[k] for k in keys)))


					keys = ["L","Omega","Delta","D"]

					if "Ef" not in database[grp]:
						job = (grp,"Ef",tuple(pars[k] for k in keys))
						if job not in jobs:
							jobs.append(job)
					else:
						job = (grp,"Ef",tuple(pars[k] for k in keys))
						if job not in jobs:
							jobs.append(job)

						del database[grp]["Ef"]

				
				# for grp,name,args in jobs:
				# 	if "V" in name:
				# 		r = Floquet_probe(*args)
				# 	elif "Ef" in name:
				# 		r = Floquet_eigenvalues(*args)
				# 	print(r.shape)

				niter = 0
				job_iter = iter(jobs)
				while(niter < len(jobs)):
					future_to_ind = {}
					for njobs in range(10*comm.size):
						grp,name,args = next(job_iter); niter += 1
						if "V" in name:
							future = executor.submit(Floquet_probe,*args)
						elif "Ef" in name:
							future = executor.submit(Floquet_eigenvalues,*args)

						future_to_ind[future] = (grp,name)

						if niter >= len(jobs):
							break

					for future in as_completed(future_to_ind):
						grp,name = future_to_ind[future]
						result,*other = future.result()
						database[grp].create_dataset(name,data=result,dtype=result.dtype)

						print("job {}/{} completed".format(grp,name))
						sys.stdout.flush()

						database.flush()




if __name__ == '__main__':
	main()







