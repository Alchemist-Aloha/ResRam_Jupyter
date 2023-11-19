
def main():

	import numpy as np
	import time
	import os
	import tdwp 
	import params as p
	import sys
	import matplotlib.pyplot as plt
	        
	abs_cross,fl_cross,raman_cross,boltz_states,boltz_coef  = tdwp.cross_sections(p.convEL,p.delta,p.theta,p.D,p.L,p.M,p.E0)

	raman_spec = np.zeros((len(p.rshift),len(p.rpumps)))
	for i in range(len(p.rpumps)):
		rp = min(range(len(p.convEL)),key=lambda j:abs(p.convEL[j]-p.rpumps[i]))
		for l in np.arange(len(p.wg)):
			raman_spec[:,i] += np.real((raman_cross[l,rp]))*(1/np.pi)*(0.5*p.res)/((p.rshift-p.wg[l])**2+(0.5*p.res)**2)

	raman_full = np.zeros((len(p.convEL),len(p.rshift)))
	for i in range(len(p.convEL)):
		for l in np.arange(len(p.wg)):
			raman_full[i,:] += np.real((raman_cross[l,i]))*(1/np.pi)*(0.5*p.res)/((p.rshift-p.wg[l])**2+(0.5*p.res)**2)

	# plt.contour(raman_full)
	# plt.show()


	if any([i == 'data' for i in os.listdir('./')]) == True:
		pass
	else:
		os.mkdir('./data')
            
	np.savetxt("data/profs.dat",np.array(np.real(raman_cross)))
	           
	np.set_printoptions(threshold=sys.maxsize)	                
	np.savetxt("data/profs.dat",np.real(np.transpose(raman_cross)))
	np.savetxt("data/raman_spec.dat",raman_spec)
	np.savetxt("data/EL.dat",p.convEL)	
	np.savetxt("data/Abs.dat",np.real(abs_cross))	
	np.savetxt("data/Fl.dat",np.real(fl_cross))
	#np.savetxt("data/Disp.dat",np.real(disp_cross))
	np.savetxt("data/rshift.dat",p.rshift)

	with open("data/output.txt",'w') as o:
		o.write("E00 = "),o.write(str(p.E0)), o.write(" cm-1 \n")
		o.write("gamma = "),o.write(str(p.gamma)), o.write(" cm-1 \n")
		o.write("theta = "),o.write(str(p.theta)), o.write(" cm-1 \n")
		o.write("M = "),o.write(str(p.M)),o.write(" Angstroms \n")
		o.write("n = "),o.write(str(p.n)),o.write("\n")
		o.write("T = "),o.write(str(p.T)),o.write(" Kelvin \n")
		o.write("solvent reorganization energy = "), o.write(str(p.s_reorg)), o.write(" cm-1 \n")
		o.write("internal reorganization energy = "),o.write(str(p.w_reorg)),o.write(" cm-1 \n")
		o.write("reorganization energy = "),o.write(str(p.reorg)),o.write(" cm-1 \n\n")
		o.write("Boltzmann averaged states and their corresponding weights \n")
		o.write(str(boltz_coef)),o.write("\n") 
		o.write(str(boltz_states)),o.write("\n") 
		
	o.close()


