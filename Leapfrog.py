

def Leapfrog(z,h,f):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

	dim = int(len(z)/2)

	z[dim:] = z[dim:]+h/2*f(z[:dim])
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*f(z[:dim])

	return z
