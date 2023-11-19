import warnings
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import lmfit

np.set_printoptions(threshold=sys.maxsize)

warnings.filterwarnings('ignore') # Supresses 'casting to real discards complex part' warning

def load(freqs,deltas,rpumps_inp,inp_inp):
    global wg,we,delta,convEL,theta,D,L,M,E0,th,EL,tm,tp,E0_range,gamma,k,beta,eta,S,order,Q,preA,preF,preR,boltz_coef,rshift,b,rpumps,res,n,T,s_reorg,w_reorg,reorg,convergence,inp
    wg = np.asarray(np.loadtxt(freqs)) # Ground state normal mode frequencies cm^-1 
    we = np.asarray(np.loadtxt(freqs)) # Excited state normal mode frequencies cm^-1
    delta = np.asarray(np.loadtxt(deltas)) # Dimensionless displacements 
    #print(delta)
    S = (delta**2)/2
    
    with open(inp_inp,'r') as i: #loading inp.txt
        
        inp = i.readlines()

        j=0
        for l in inp:
            l = l.partition('#')[0]
            l = l.rstrip()
            inp[j] = l
            j+=1
        
        hbar =  5.3088 # plancks constant cm^-1*ps
        T = float(inp[13]) # Temperature K
        kbT = 0.695*T # kbT energy (cm^-1/K)*cm^-1=cm^-1
        cutoff = kbT*0.1 # cutoff for boltzmann dist in wavenumbers
        if T > 10.0:
            beta = 1/kbT # beta cm
            eta = 1/(np.exp(wg/kbT)-1) # array of average thermal occupation numbers for each mode
        elif T < 10.0:
            beta = 1/kbT
            #beta = float("inf")
            eta = np.zeros(len(wg))

        gamma = float(inp[0]) # Homogeneous broadening parameter cm^-1
        theta = float(inp[1]) # Static inhomogenous broadening parameter cm^-1
        E0 = float(inp[2]) # E0 cm^-1

        ## Brownian Oscillator parameters ##
        k = float(inp[3]) # kappa parameter
        D =  gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k) # D parameter 
        L =  k*D # LAMBDA parameter

        s_reorg = beta*(L/k)**2/2 # reorganization energy cm^-1
        w_reorg = 0.5*np.sum((delta)**2*wg) # internal reorganization energy
        reorg =  w_reorg + s_reorg # Total reorganization energy

        ## Time and energy range stuff ##
        ts = float(inp[4]) # Time step (ps)
        ntime = float(inp[5]) #175 # ntime steps
        UB_time = ntime*ts # Upper bound in time range
        t = np.linspace(0,UB_time,int(ntime)) # time range in ps
        EL_reach = float(inp[6]) # How far plus and minus E0 you want 
        EL = np.linspace(E0-EL_reach,E0+EL_reach,1000) # range for spectra cm^-1	
        E0_range = np.linspace(-EL_reach*0.5,EL_reach*0.5,501)# static inhomogeneous convolution range	

        th = np.array(t/hbar) # t/hbar 

        ntime_rot = ntime/np.sqrt(2)
        ts_rot = ts/np.sqrt(2)
        UB_time_rot = ntime_rot*ts_rot
        tp = np.linspace(0,UB_time_rot,int(ntime_rot)) 
        tm = None
        tm = np.append(-np.flip(tp[1:],axis=0),tp)
        convEL = np.linspace(E0-EL_reach*0.5,E0+EL_reach*0.5,(max(len(E0_range),len(EL))-min(len(E0_range),len(EL))+1)) # Excitation axis after convolution with inhomogeneous distribution

        M = float(inp[7]) # Transition dipole length angstroms
        n = float(inp[8]) # Refractive index

        rpumps = np.asarray(np.loadtxt(rpumps_inp)) # Raman pump wavelengths to compute spectra at
        rshift = np.arange(float(inp[9]),float(inp[10]),float(inp[11])) # range and step size of Raman spectrum
        res = float(inp[12]) # Peak width in Raman spectra

        # Determine order from Boltzmann distribution of possible initial states #
        convergence = float(inp[14]) # desired boltzmann coefficient for cutoff
        boltz_toggle = int(inp[15])

        if boltz_toggle == 1:
            boltz_states,boltz_coef,dos_energy = boltz_states()
            if T == 0.0:
                state = 0
            else:
                state = min(range(len(boltz_coef)),key=lambda j:abs(boltz_coef[j]-convergence))

            if state == 0:
                order = 1
            else:
                order = max(max(boltz_states[:state])) + 1
        if boltz_toggle == 0:
            boltz_states,boltz_coef,dos_energy = [0,0,0]
            order = 1

        a = np.arange(order)
        b = a
        Q = np.identity(len(wg),dtype=int)
        
        #wq = None
        #wq = np.append(wg,wg)
    i.close()
    ## Prefactors for absorption and Raman cross-sections ##
    if order == 1:
        preR = 2.08e-20*(ts**2) #(0.3/pi) puts it in differential cross section 
    elif order > 1:
        preR = 2.08e-20*(ts_rot**2)

    preA = ((5.744e-3)/n)*ts
    preF = preA*n**2    

def boltz_states():
    wg = wg.astype(int)
    cutoff = range(int(cutoff))
    dos = range(len(cutoff))
    states = []
    dos_energy = []
    

    def count_combs(left, i, comb, add):
        if add: comb.append(add)
        if left == 0 or (i+1) == len(wg):
            if (i+1) == len(wg) and left > 0:
                if left % wg[i]: # can't get the exact score with this kind of wg
                    return 0         # so give up on this recursive branch
                comb.append( (left/wg[i], wg[i]) ) # fix the amount here
                i += 1
            while i < len(wg):
                comb.append( (0, wg[i]) )
                i += 1
            states.append([x[0] for x in comb])
            return 1
        cur = wg[i]
        return sum(count_combs(left-x*cur, i+1, comb[:], (x,cur)) for x in range(0, int(left/cur)+1))
    
    boltz_dist = [] #np.zeros(len(dos))
    for i in range(len(cutoff)):
        dos[i] = count_combs(cutoff[i], 0, [], None)
        if dos[i] > 0.0:
            boltz_dist.append([np.exp(-cutoff[i]*beta)])
            dos_energy.append(cutoff[i])
    

    norm = np.sum(boltz_dist)

    np.reshape(states,-1,len(cutoff))

    return states,boltz_dist/norm,dos_energy



def g(t):
    D =  gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k) # D parameter 
    L =  k*D # LAMBDA parameter
    g = ((D/L)**2)*(L*t-1+np.exp(-L*t))+1j*((beta*D**2)/(2*L))*(1-np.exp(-L*t)) 
    #g = p.gamma*np.abs(t)#
    return g

def A(t):
    #K=np.zeros((len(p.wg),len(t)),dtype=complex)

    if type(t) == np.ndarray:
        K = np.zeros((len(wg),len(th)),dtype=complex)
    else:
        K=np.zeros((len(wg),1),dtype=complex) 
    for l in np.arange(len(wg)):
        K[l,:] = (1+eta[l])*S[l]*(1-np.exp(-1j*wg[l]*t))+eta[l]*S[l]*(1-np.exp(1j*wg[l]*t))
    A = M**2*np.exp(-np.sum(K,axis=0))
    return A

def R(t1,t2):
    Ra = np.zeros((len(a),len(wg),len(wg),len(EL)),dtype=complex)
    R = np.zeros((len(wg),len(wg),len(EL)),dtype=complex)
    # for l in np.arange(len(p.wg)):
    # 	for q in p.Q:
    for idxq,q in enumerate(Q,start=0):
        for idxl,l in enumerate(q,start=0):

            wg = wg[idxl]
            S = S[idxl]
            eta = eta[idxl]
            if l == 0:
                for idxa,a in enumerate(a,start=0):
                    Ra[idxa,idxq,idxl,:] = ((1./factorial(a))**2)*((eta*(1+eta))**a)*S**(2*a)*(((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1))))*((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1)))))**a
                R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
            elif l > 0:
                for idxa,a in enumerate(a[l:],start=0):
                    Ra[idxa,idxq,idxl,:] = ((1./(factorial(a)*factorial(a-l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(1-np.exp(1j*wg*t2)))**a)*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a-l)
                R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
            elif l < 0:
                for idxa,a in enumerate(b[-l:],start=0):
                    Ra[idxa,idxq,idxl,:] = ((1./(factorial(a)*factorial(a+l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(1-np.exp(1j*wg*t2)))**(a+l))*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a)
                R[idxq,idxl,:] = np.sum(Ra[:,idxq,idxl,:],axis=0)
    return np.prod(R,axis=1)	

def cross_sections(convEL,delta,theta,D,L,M,E0): 
    global tm, EL, wg,abs_cross,fl_cross,raman_cross,boltz_states,boltz_coef
    q_r = np.ones((len(wg),len(wg),len(th)),dtype=complex)
    K_r = np.zeros((len(wg),len(EL),len(th)),dtype=complex)
    # elif p.order > 1:
    # 	K_r = np.zeros((len(p.tm),len(p.tp),len(p.wg),len(p.EL)),dtype=complex)
    integ_r1 = np.zeros((len(tm),len(EL)),dtype=complex)
    integ_r = np.zeros((len(wg),len(EL)),dtype=complex)
    raman_cross = np.zeros((len(wg),len(convEL)),dtype=complex)

    if theta == 0.0:
        H = 1. #np.ones(len(p.E0_range))
    else:
        H = (1/(theta*np.sqrt(2*np.pi)))*np.exp(-((E0_range)**2)/(2*theta**2))

    thth,ELEL = np.meshgrid(th,EL,sparse=True)



    K_a = np.exp(1j*(ELEL-(E0))*thth-g(thth))*A(thth)
    K_f = np.exp(1j*(ELEL-(E0))*thth-np.conj(g(thth)))*np.conj(A(thth))

    ## If the order desired is 1 use the simple first order approximation ##
    if order == 1:
        for idxq,q in enumerate(Q,start=0):
            for idxl,l in enumerate(q,start=0):
                if q[idxl] > 0:
                    q_r[idxq,idxl,:] = (1./factorial(q[idxl]))**(0.5)*(((1+eta[idxl])**(0.5)*delta[idxl])/np.sqrt(2))**(q[idxl])*(1-np.exp(-1j*wg[idxl]*thth))**(q[idxl])
                elif q[idxl] < 0:
                    q_r[idxq,idxl,:] = (1./factorial(np.abs(q[idxl])))**(0.5)*(((eta[l])**(0.5)*delta[l])/np.sqrt(2))**(-q[idxl])*(1-np.exp(1j*wg[idxl]*thth))**(-q[idxl])
            K_r[idxq,:,:] = K_a*(np.prod(q_r,axis=1)[idxq])

    ## If the order is greater than 1, carry out the sums R and compute the full double integral
    ##### Higher order is still broken, need to fix #####
    elif order > 1:

        tpp,tmm,ELEL = np.meshgrid(tp,tm,EL,sparse=True)
        K_r = np.exp(1j*(ELEL-E0)*np.sqrt(2)*tmm-g(tpp+tmm)/(np.sqrt(2))-np.conj(g((tpp-tmm)/(np.sqrt(2)))))#*A((tpp+tmm)/(np.sqrt(2)))*np.conj(A((tpp-tmm)/(np.sqrt(2))))#*R((tpp+tmm)/(np.sqrt(2)),(tpp-tmm)/(np.sqrt(2)))

        for idxtm,tm in enumerate(tm,start=0):
            integ_r1[idxtm,:] = np.trapz(K_r[(np.abs(len(tm)/2-idxtm)):,idxtm,:],axis=0)

        integ = np.trapz(integ_r1,axis=0)
    ######################################################

    integ_a = np.trapz(K_a,axis=1)
    abs_cross = preA*convEL*np.convolve(integ_a,np.real(H),'valid')/(np.sum(H))

    integ_f = np.trapz(K_f,axis=1)
    fl_cross = preF*convEL*np.convolve(integ_f,np.real(H),'valid')/(np.sum(H))

    # plt.plot(p.convEL,abs_cross)
    # plt.plot(p.convEL,fl_cross)
    # plt.show()


    # plt.plot(integ_a)
    # plt.plot(integ_f)
    # plt.show()
    #print p.s_reorg
    #print p.w_reorg
    #print p.reorg

    for l in range(len(wg)):
        if order == 1:
            integ_r[l,:] = np.trapz(K_r[l,:,:],axis=1)
            raman_cross[l,:] = preR*convEL*(convEL-wg[l])**3*np.convolve(integ_r[l,:]*np.conj(integ_r[l,:]),np.real(H),'valid')/(np.sum(H))
        elif order > 1:
            raman_cross[l,:] = preR*convEL*(convEL-wg[l])**3*np.convolve(integ_r[l,:],np.real(H),'valid')/(np.sum(H))

    # plt.plot(p.convEL,fl_cross)
    # plt.plot(p.convEL,abs_cross)
    # plt.show()

    # plt.plot(p.convEL,raman_cross[0])
    # plt.plot(p.convEL,raman_cross[1])
    # plt.plot(p.convEL,raman_cross[2])
    # plt.plot(p.convEL,raman_cross[3])
    # plt.show()
    # exit()


    return (abs_cross,fl_cross,raman_cross,boltz_states,boltz_coef)

def main():
    global raman_cross, abs_cross,fl_cross,boltz_coef,boltz_states
    abs_cross,fl_cross,raman_cross,boltz_states,boltz_coef  = cross_sections(convEL,delta,theta,D,L,M,E0)
    raman_spec = np.zeros((len(rshift),len(rpumps)))
    print(rpumps)
    for i in range(len(rpumps)):
        #rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
        min_diff = float('inf')
        min_index = None

        for j in range(len(convEL)):
            diff = np.absolute(convEL[j] - rpumps[i])
            if diff < min_diff:
                min_diff = diff
                min_index = j

        rp = min_index
        #print(rp)
        #print(rpumps)
        for l in np.arange(len(wg)):
            raman_spec[:,i] += np.real((raman_cross[l,rp]))*(1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)

    raman_full = np.zeros((len(convEL),len(rshift)))
    for i in range(len(convEL)):
        for l in np.arange(len(wg)):
            raman_full[i,:] += np.real((raman_cross[l,i]))*(1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)

    # plt.contour(raman_full)
    # plt.show()


    if any([i == 'data' for i in os.listdir('./')]) == True:
        pass
    else:
        os.mkdir('./data')
            
    np.savetxt("data/profs.dat",np.array(np.real(raman_cross)))
               
    np.set_printoptions(threshold=sys.maxsize)	                
    np.savetxt("data/profs.dat",np.real(np.transpose(raman_cross)),delimiter = "\t")
    np.savetxt("data/raman_spec.dat",raman_spec,delimiter = "\t")
    np.savetxt("data/EL.dat",convEL)	
    np.savetxt("data/Abs.dat",np.real(abs_cross))	
    np.savetxt("data/Fl.dat",np.real(fl_cross))
    #np.savetxt("data/Disp.dat",np.real(disp_cross))
    np.savetxt("data/rshift.dat",rshift)

    with open("data/output.txt",'w') as o:
        o.write("E00 = "),o.write(str(E0)), o.write(" cm-1 \n")
        o.write("gamma = "),o.write(str(gamma)), o.write(" cm-1 \n")
        o.write("theta = "),o.write(str(theta)), o.write(" cm-1 \n")
        o.write("M = "),o.write(str(M)),o.write(" Angstroms \n")
        o.write("n = "),o.write(str(n)),o.write("\n")
        o.write("T = "),o.write(str(T)),o.write(" Kelvin \n")
        o.write("solvent reorganization energy = "), o.write(str(s_reorg)), o.write(" cm-1 \n")
        o.write("internal reorganization energy = "),o.write(str(w_reorg)),o.write(" cm-1 \n")
        o.write("reorganization energy = "),o.write(str(reorg)),o.write(" cm-1 \n\n")
        o.write("Boltzmann averaged states and their corresponding weights \n")
        o.write(str(boltz_coef)),o.write("\n") 
        o.write(str(boltz_states)),o.write("\n") 
        
    o.close()


def raman_residual(param):
    global rcross_exp, gamma, M,abs_exp
    #load("freqs.dat","deltas.dat","rpumps.dat","inp.txt")
    for i in range(len(delta)):
        delta[i] = param.valuesdict()['delta'+str(i)]
    gamma = param.valuesdict()['gamma']
    M = param.valuesdict()['transition_length']
    k = float(inp[3]) # kappa parameter
    D =  gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k) # D parameter 
    L =  k*D # LAMBDA parameter
    print(delta)
    abs_cross,fl_cross,raman_cross,boltz_states,boltz_coef  = cross_sections(convEL,delta,theta,D,L,M,E0)
    correlation = (np.corrcoef(np.real(abs_cross), abs_exp)[0, 1])
    print("Correlation between exp and sim absorption is "+ str(correlation))
    correlation = -100*correlation #Minimize the negative correlation to get better fit
    #rcross_exp = np.asarray(np.loadtxt("rshift_exp.dat"))
    if rcross_exp.ndim == 1:    #Convert 1D array to 2D
        rcross_exp = np.reshape(rcross_exp,(-1,1))
        #print("Raman cross section expt is converted to a 2D array")
    sigma = np.zeros_like(delta)
    for i in range(len(rpumps)):
    #rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
        min_diff = float('inf')
        rp = None

        for j in range(len(convEL)):
            diff = np.absolute(convEL[j] - rpumps[i])
            if diff < min_diff:
                min_diff = diff
                rp = j
        #print(rp)
        for j in range(len(wg)):
            #print(j,i)
            sigma[j] = sigma[j] + (1e8*(np.real(raman_cross[j,rp])-rcross_exp[j,i]))**2
    #sigma = np.tanh(sigma)   

    #sigma = np.log10(sigma)*-1
    
    sigma=np.append(sigma,correlation)
    print(sigma)
    total_sigma = np.sum(sigma)
    print("Total sigma is\t")
    print(total_sigma)
    return sigma

def param_init(deltas_inp,abs_inp):
    global abs_exp,rcross_exp
    delta_params = np.asarray(np.loadtxt(deltas_inp))
    abs_exp = np.asarray(np.loadtxt(abs_inp)[:,1]) #load experimental abs spec
    rcross_exp = np.asarray(np.loadtxt("rshift_exp.dat"))
    params_lmfit = lmfit.Parameters()
    for i in range(len(delta_params)):
        params_lmfit.add('delta'+str(i),value=delta_params[i],min=0.0,max=1.0)
    return params_lmfit