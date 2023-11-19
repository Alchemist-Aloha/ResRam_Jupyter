import numpy as np
import params as p
import matplotlib.pyplot as plt


def boltz_states():
    wg = p.wg.astype(int)
    cutoff = range(int(p.cutoff))
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
            boltz_dist.append([np.exp(-cutoff[i]*p.beta)])
            dos_energy.append(cutoff[i])
    

    norm = np.sum(boltz_dist)

    np.reshape(states,-1,len(cutoff))

    return states,boltz_dist/norm,dos_energy
