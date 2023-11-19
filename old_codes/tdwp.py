import warnings
from math import factorial
import numpy as np
import params as p
import boltz_states as bs
import ovlp_fn as fn
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

warnings.filterwarnings('ignore') # Supresses 'casting to real discards complex part' warning


def g(t):
	D =  p.gamma*(1+0.85*p.k+0.88*p.k**2)/(2.355+1.76*p.k) # D parameter 
	L =  p.k*D # LAMBDA parameter
	g = ((D/L)**2)*(L*t-1+np.exp(-L*t))+1j*((p.beta*D**2)/(2*L))*(1-np.exp(-L*t)) 
	#g = p.gamma*np.abs(t)#
	return g

def A(t):
	#K=np.zeros((len(p.wg),len(t)),dtype=complex)

	if type(t) == np.ndarray:
		K = np.zeros((len(p.wg),len(p.th)),dtype=complex)
	else:
		K=np.zeros((len(p.wg),1),dtype=complex) 
	for l in np.arange(len(p.wg)):
		K[l,:] = (1+p.eta[l])*p.S[l]*(1-np.exp(-1j*p.wg[l]*t))+p.eta[l]*p.S[l]*(1-np.exp(1j*p.wg[l]*t))
	A = p.M**2*np.exp(-np.sum(K,axis=0))
	return A

def R(t1,t2):
	Ra = np.zeros((len(p.a),len(p.wg),len(p.wg),len(p.EL)),dtype=complex)
	R = np.zeros((len(p.wg),len(p.wg),len(p.EL)),dtype=complex)
	# for l in np.arange(len(p.wg)):
	# 	for q in p.Q:
	for idxq,q in enumerate(p.Q,start=0):
		for idxl,l in enumerate(q,start=0):

			wg = p.wg[idxl]
			S = p.S[idxl]
			eta = p.eta[idxl]
			if l == 0:
				for idxa,a in enumerate(p.a,start=0):
					Ra[idxa,idxq,idxl,:] = ((1./factorial(a))**2)*((eta*(1+eta))**a)*S**(2*a)*(((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1))))*((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1)))))**a
				R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
			elif l > 0:
				for idxa,a in enumerate(p.a[l:],start=0):
					Ra[idxa,idxq,idxl,:] = ((1./(factorial(a)*factorial(a-l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(1-np.exp(1j*wg*t2)))**a)*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a-l)
				R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
			elif l < 0:
				for idxa,a in enumerate(p.b[-l:],start=0):
					Ra[idxa,idxq,idxl,:] = ((1./(factorial(a)*factorial(a+l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(1-np.exp(1j*wg*t2)))**(a+l))*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a)
				R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
	return np.prod(R,axis=1)	

def cross_sections(convEL,delta,theta,D,L,M,E0): 

	q_r = np.ones((len(p.wg),len(p.wg),len(p.th)),dtype=complex)
	K_r = np.zeros((len(p.wg),len(p.EL),len(p.th)),dtype=complex)
	# elif p.order > 1:
	# 	K_r = np.zeros((len(p.tm),len(p.tp),len(p.wg),len(p.EL)),dtype=complex)
	integ_r1 = np.zeros((len(p.tm),len(p.EL)),dtype=complex)
	integ_r = np.zeros((len(p.wg),len(p.EL)),dtype=complex)
	raman_cross = np.zeros((len(p.wg),len(p.convEL)),dtype=complex)

	if theta == 0.0:
		H = 1. #np.ones(len(p.E0_range))
	else:
		H = (1/(theta*np.sqrt(2*np.pi)))*np.exp(-((p.E0_range)**2)/(2*theta**2))

	thth,ELEL = np.meshgrid(p.th,p.EL,sparse=True)



	K_a = np.exp(1j*(ELEL-(p.E0))*thth-g(thth))*A(thth)
	K_f = np.exp(1j*(ELEL-(p.E0))*thth-np.conj(g(thth)))*np.conj(A(thth))

	## If the order desired is 1 use the simple first order approximation ##
	if p.order == 1:
		for idxq,q in enumerate(p.Q,start=0):
			for idxl,l in enumerate(q,start=0):
				if q[idxl] > 0:
					q_r[idxq,idxl,:] = (1./factorial(q[idxl]))**(0.5)*(((1+p.eta[idxl])**(0.5)*delta[idxl])/np.sqrt(2))**(q[idxl])*(1-np.exp(-1j*p.wg[idxl]*thth))**(q[idxl])
				elif q[idxl] < 0:
					q_r[idxq,idxl,:] = (1./factorial(np.abs(q[idxl])))**(0.5)*(((p.eta[l])**(0.5)*delta[l])/np.sqrt(2))**(-q[idxl])*(1-np.exp(1j*p.wg[idxl]*thth))**(-q[idxl])
			K_r[idxq,:,:] = K_a*(np.prod(q_r,axis=1)[idxq])

	## If the order is greater than 1, carry out the sums R and compute the full double integral
	##### Higher order is still broken, need to fix #####
	elif p.order > 1:

		tpp,tmm,ELEL = np.meshgrid(p.tp,p.tm,p.EL,sparse=True)
		K_r = np.exp(1j*(ELEL-p.E0)*np.sqrt(2)*tmm-g(tpp+tmm)/(np.sqrt(2))-np.conj(g((tpp-tmm)/(np.sqrt(2)))))#*A((tpp+tmm)/(np.sqrt(2)))*np.conj(A((tpp-tmm)/(np.sqrt(2))))#*R((tpp+tmm)/(np.sqrt(2)),(tpp-tmm)/(np.sqrt(2)))

		for idxtm,tm in enumerate(p.tm,start=0):
			integ_r1[idxtm,:] = np.trapz(K_r[(np.abs(len(p.tm)/2-idxtm)):,idxtm,:],axis=0)

		integ = np.trapz(integ_r1,axis=0)
	######################################################

	integ_a = np.trapz(K_a,axis=1)
	abs_cross = p.preA*p.convEL*np.convolve(integ_a,np.real(H),'valid')/(np.sum(H))

	integ_f = np.trapz(K_f,axis=1)
	fl_cross = p.preF*p.convEL*np.convolve(integ_f,np.real(H),'valid')/(np.sum(H))

	# plt.plot(p.convEL,abs_cross)
	# plt.plot(p.convEL,fl_cross)
	# plt.show()


	# plt.plot(integ_a)
	# plt.plot(integ_f)
	# plt.show()
	#print p.s_reorg
	#print p.w_reorg
	#print p.reorg

	for l in range(len(p.wg)):
		if p.order == 1:
			integ_r[l,:] = np.trapz(K_r[l,:,:],axis=1)
			raman_cross[l,:] = p.preR*p.convEL*(p.convEL-p.wg[l])**3*np.convolve(integ_r[l,:]*np.conj(integ_r[l,:]),np.real(H),'valid')/(np.sum(H))
		elif p.order > 1:
			raman_cross[l,:] = p.preR*p.convEL*(p.convEL-p.wg[l])**3*np.convolve(integ_r[l,:],np.real(H),'valid')/(np.sum(H))

	# plt.plot(p.convEL,fl_cross)
	# plt.plot(p.convEL,abs_cross)
	# plt.show()

	# plt.plot(p.convEL,raman_cross[0])
	# plt.plot(p.convEL,raman_cross[1])
	# plt.plot(p.convEL,raman_cross[2])
	# plt.plot(p.convEL,raman_cross[3])
	# plt.show()
	# exit()


	return (abs_cross,fl_cross,raman_cross,p.boltz_states,p.boltz_coef)









