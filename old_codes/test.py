import numpy as np

def boltz_states():

	cutoff = 41#int(p.cutoff) # N
	wg = [4,6,10] #p.wg.astype(int) # C 
	s = np.zeros(len(wg))


	P = np.zeros(cutoff) # P
	
	for i in range(len(wg)): # len(wg) K
		j = wg[i]
		jp1 = j + 1
		P[j] = P[j] + 1
		
		for m in range(jp1,cutoff):
			mmj = m - j
			P[m] = P[m] + P[mmj]
	print P

boltz_states()

cents = 42
points = [4,6,10] #[25,10,5,1]

def fill_mat():
	m = []
	def count_combs(left, i, comb, add):
	    if add: comb.append(add)
	    if left == 0 or (i+1) == len(points):
	        if (i+1) == len(points) and left > 0:
	            if left % points[i]: # can't get the exact score with this kind of points
	                return 0         # so give up on this recursive branch
	            comb.append( (left/points[i], points[i]) ) # fix the amount here
	            i += 1
	        while i < len(points):
	            comb.append( (0, points[i]) )
	            i += 1
	        m.append([x[0] for x in comb])
	        return 1
	    cur = points[i]
	    return sum(count_combs(left-x*cur, i+1, comb[:], (x,cur)) for x in range(0, int(left/cur)+1))
	count = count_combs(cents, 0, [], None)

	return m, count

print fill_mat()






