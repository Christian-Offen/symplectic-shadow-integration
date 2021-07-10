import numpy as np
from scipy.linalg import cho_factor, cho_solve

class ShadowMidpoint():
	
	def classicInt(self,z,f,h,verbose = False):
	## classical midpoint rule
	

		fstage = lambda stg: h * f(z+0.5*stg)

		# fixed point iterations to compute stages

		stageold=0.
		stage = fstage(stageold) +0.
		Iter = 0

		while np.amax(abs(stage - stageold)) > 1e-8 and Iter<100:
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1

		if verbose == True:
			print('Midpoint rule fixpoint iterations: ' + str(Iter) + ' Residuum: ' + str(abs(stage - stageold)))

		return z + stage



	def classicTrajectory(self,z,f,h,N=1,verbose=False):
	## trajectory computed with classicInt
	
		trj = np.zeros((len(z),N+1))
		trj[:,0] = z.copy()

		for j in range(0,N):
			trj[:,j+1] = self.classicInt(trj[:,j].copy(),f,h,verbose)
		
		return trj
		
		
	def train(self,trainin, trainout, h, k=False, dk=False, ddk=False, x0=0, H0=0, sigma=1e-13):
	# fit GP to training data 
	# with kernel k and derivatives dk, Hessian ddk (set as constant zero matrix if it does not exist)
	# if k is not specified, we use radial basis functions
	# normalisation H0 at x0
	# using Tikhonov regularisation with parameter sigma
	
	
		if k==False:
		# use radial basis functions if not specified otherwise
			
			# kernel function and its gradient wrt. one input
			e   = 2.           # length scale
			k   = lambda x,y: np.exp(-1/(e**2)*np.linalg.norm(x-y)**2)
			dk  = lambda x,y: 2/e**2*(x-y)*k(x,y)
			ddk = lambda x,y: 2/e**2*k(x,y)*(-np.identity(len(x)) + 2/e**2*np.outer(x-y,x-y))
		
	
		dim = int(len(trainin)/2)
	
		J=np.block([    [np.zeros((dim,dim)),-np.identity(dim)],    [np.identity(dim),np.zeros((dim,dim))] ])
		Jinv = -J

		# Inverse modified vector field for Symplectic Euler
		g=np.hstack((J @ ((trainout-trainin)/h)).transpose())
	

		# data points for inference compatible with Symplectic Euler X=(Q,p)
		X = (trainin + trainout)/2
		X = X.transpose()

		# normalisation value H0 at H(x0)
		x0 = x0*np.ones(2*dim) # make sure, x0 has the correct size if it is not set explicitly
		RHS = np.append(g,H0)
	
		Y=X
	
		print("Start training with for "+str(len(Y))+ " data points")

		# Covariance matrix for the multivariate normal distribution of the random vector (H(Y[0]),H(Y[1]),...,H(Y[-1]))

		print("Start computation of covariance matrix.")
		K = np.empty((Y.shape[0],Y.shape[0]))
		for i in range(0,Y.shape[0]):
			K[i,i] = k(Y[i],Y[i])
			for j in range(i+1,Y.shape[0]):
				K[i,j] = k(Y[i],Y[j])
				K[j,i] = K[i,j]
		print("Covariance matrix of shape "+str(K.shape)+"computed.")

		# Tikhonov regularisation and Cholesky decomposition
		K = K + sigma*np.identity(K.shape[0])
		print("Start Cholesky decomposition of "+str(K.shape)+" Matrix")
		L, low = cho_factor(K)
		print("Cholesky decomposition completed.")

		# creation of linear systems to compute mean of inverse modified Hamiltonian from inverse modified vectorfield
		print("Create LHS of linear system for H at test points.")
		dkY = lambda x : [dk(x,y) for y in Y ]
		dKK = [cho_solve((L, low), np.array(dkY(xi))).transpose() for xi in X ]       # np.array(dkY(xi)).transpose() @ Kinv
		dKK=np.vstack(dKK)

		# normalisation of H
		kY = [k(x0,y) for y in Y]
		kYY=cho_solve((L, low), np.array(kY)).transpose()

		LHS = np.vstack([dKK,kYY])
		print("Creation of linear system completed.")

		# solution of linear system
		print("Solve least square problem of dimension "+str(LHS.shape))
		HY,res,rank, _ = np.linalg.lstsq(LHS,RHS,rcond=None)
		
		# provide data for other methods	
		self.h = h
		self.Jinv = Jinv
		self.k = k
		self.dk = dk
		self.ddk = ddk
		self.HX = HY
		self.X=X
		self.L = L
		self.KinvH = cho_solve((L,low),HY)

		return res, rank

	# mean values of inverse modified Hamiltonian and its jet at x infered from values at (X,HX)
	
	def invmodH(self,x):

		kX = [self.k(x,y) for y in self.X ]
		return np.array(kX).transpose() @ self.KinvH  # H
	
	def invmoddH(self,x):

		dkX = [self.dk(x,y) for y in self.X ]
		return np.array(dkX).transpose() @ self.KinvH  # grad H
	
	
	def invmodddH(self,x):

		ddkX = [self.ddk(x,y) for y in self.X ]

		return np.array(ddkX).transpose() @ self.KinvH  # Hessian H

	
	def predictMotion(self,z0,N,verbose=False):
	## apply classical integrator to inverse modified vector field
		
		f = lambda z: self.Jinv @ self.invmoddH(z)
		return self.classicTrajectory(z0,f,self.h,N,verbose)
		
		
		
	def HRecover(self,x):
    
		HH = self.invmodH(x)
		grad = self.invmoddH(x)
		Jg = self.Jinv @ grad
		hess = self.invmodddH(x)

		H1 = HH
		H2 = H1 - self.h**2*1/24*(Jg @ hess @ Jg)

		return HH,H1,H2	


