from scipy.special import comb
import numpy as np
from math import factorial
from math import floor

def hermite(n,z):
    if n==0:
        return 1
    elif n==1:
        return 2*z
    else:
        return 2*z*hermite(n-1,z)-2*(n-1)*hermite(n-2,z)

def ovlp_fn(delta,wg,we,th,f,n):

	wp = we + wg
	wm = we - wg
	ks = int(floor((f+n)/2.))
	
	gam = (2*(we/wg)+1j*((we/wg)**2-1)*np.sin(we*th))/(2*(we/wg)-1j*((we/wg)**2-1)*np.sin(we*th))

	a = ((wm-wp*np.exp(-1j*we*th))/(2.0*wp-2.0*wm*np.exp(-1j*we*th)))**(0.5)

	func = -(wg*(1-np.exp(-1j*we*th)))/(wp-2*wm*np.exp(-1j*we*th))

	psi = (wp**2/(4*we*wg))*(1-(wm/wp)**2*np.exp(-2*1j*we*th))

	herm = np.zeros(len(th),dtype=complex)
	efnk = np.zeros(len(th),dtype=complex)
	for k in range(ks+1):
		for q in range(2*k+1):
			efnk = efnk + (-1.0)**q*comb(f,2.0*k-q)*comb(n,q) 
		herm = herm + (factorial(2.*k)/factorial(k))*efnk*gam**k*hermite(f+n-2*k,(we/wg)**(0.5)*func*delta/a)

	ovlp_fn = (factorial(f)*factorial(n)*2.0**(f+n)*psi)**(-0.5)*np.exp(delta**2*func)*a**(f+n)*herm

	return ovlp_fn