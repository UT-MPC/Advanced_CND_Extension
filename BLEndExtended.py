import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad
import collections
from scipy.optimize import fsolve

b = 3.3 / 3         # advertisement duration in ms, or minimum TX/RX overlapping, lambda in Ruth
max_slop = 10       # the maximum random slop added by BLE
I_tx = 2.64         # TX current draw, in mA
I_rx = 2.108        # RX current draw, in mA
I_aWP = 0.339       # warmup period from scan to advertising, in mA
I_sWP = 0.406       # warmup period from advertising to scan, in mA
I_idle = 0.339      # IDLE time between 2 beacons, in mA
battery = (225 * 3 / 2.1)  # battery capacity in mAh

A = 50

# Extended
b_sec = 0
AUX_offset = 0
B = b ### for legacy
L_Scan = A + max_slop + B 

sla_prob_list = [0.70, 0.80, 0.90]  # Desired probability of discovery
latency_list = [10000, 30000] 
sla_node_set = [3, 10, 25, 50, 75, 100]  # Set of nodes to test

n_chunk_list = [11]

# Function to compute the energy cost of one epoch given this schedule
def compute_Q(adv_interval, num_beacons, psi, E):
    idletime = E - (adv_interval + B + max_slop) - (b * num_beacons) - psi
    return (I_rx * (adv_interval + B + max_slop) + num_beacons * b * I_tx + idletime * I_idle) / E


def compute_disc_prob(A, i, E, nb, C, psiGap, minN, W_base, latency):
    Window = W_base + W_base/2

    P_oneW = compute_disc_from_lat(A, nb, C, E, psiGap, lat=Window, window=Window, W_base=W_base) 

    n = math.floor(latency/Window)
    res = (latency - Window*n)

    # Filter
    P_center = 1/Window if Window!=0 else math.inf
    if P_center*n > 1/100: #less than 1 %
        return 0

    Pd = 0
    cnt = 0
    for start in range(min(W_base, latency)):
        Pd_start = compute_disc_from_lat(A, nb, C, E, psiGap, lat=start, window=Window, W_base=W_base)

        n = math.floor((latency-start)/Window)
        res = (latency - start - Window*n)

        for i in range(minN, n+1):
            Pd += math.comb(n,i) * (P_oneW**i) * (1-P_oneW)**(n-i)

        if n>=minN-1>=0:
            P_res = compute_disc_from_lat(A, nb, C, E, psiGap, lat=res, window=Window, W_base=W_base)
            Pd += math.comb(n,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n-minN+1) * (P_res+Pd_start - Pd_start*P_res)
        
        cnt += 1

    return Pd/cnt

def compute_disc_from_lat(A, nb, C, E, psiGap, lat=0, window=0, W_base=0):

    lat = lat * (2/3) # effective W is W_base only

    k = math.floor(lat / E)
    leftover = lat - k * E
    fraction = (leftover / E)

    psi = W_base/2 
    P_in = (1-(psi+2*B)/window)

    extraBeaconRatio =  (max_slop/2) / (A + max_slop) 
    gamma = P_in * (C-2) * (1+extraBeaconRatio)

    Pnc = (1-2*B/(A+max_slop)) ** gamma
    P_in_frac = (1-(psi+2*B)/window)  * ((fraction*L_Scan-B)/(A+max_slop))
    if k>=1:
        Pd = 1 - (1-P_in*Pnc)**k * (1-P_in_frac*Pnc)
    else:
        Pd = P_in_frac*Pnc
    return Pd


for n_chunk in n_chunk_list:
    for sla_prob in sla_prob_list:
        for latency in latency_list:
            
            # Write results to output file
            out_file_name = f"/Users/hy.c/Desktop/NoD_Plus/BLEndAE/AE_blend_{sla_prob}_{latency/1000}s_log.txt"

            with open(out_file_name, 'w') as con:

                hrs = []  # Expected battery life in hours
                for sla_nodes in sla_node_set:
                    disc_best = -1
                    epoch_best = -1
                    bestA = -1
                    bestK = -1
                    bestNb = -1
                    bestW = -1

                    Q_min = np.iinfo(np.int32).max
                    I_min = np.iinfo(np.int32).max

                    for A in range(100, 101):

                        L_Scan = A + max_slop + B # extended B
                        minN = math.floor(0.8*n_chunk) + (1 if n_chunk*0.8%1>0 else 0) # mininum number of windows that needs to be captured
                        
                        for W_base in range(int(latency/(2*minN)), int(3*L_Scan), -1):
                            
                            E = W_base
                            if W_base<E:
                                break

                            if A<E/4: # base on the code in BLEnd optimizer

                                advTime = E - L_Scan
                                psiGap = advTime % (A + max_slop / 2) ## use for BLE determine last size
                                advCnt = math.floor(E/A) - 2
                                
                                disc_prob = compute_disc_prob(A, 0, E, advCnt, sla_nodes, psiGap, minN, W_base, latency)

                                if disc_prob >= sla_prob:
                                    Q = compute_Q(A, advCnt, psiGap, E)
                                    I = Q / E
                                    if I < I_min:
                                        I_min = I
                                        Q_min = Q
                                        epoch_best = E
                                        disc_best = disc_prob
                                        bestA = A
                                        bestNb = advCnt
                                        bestW = W_base

                    con.write(f"\nThe best epoch size for {sla_nodes} nodes is {epoch_best}\n")
                    con.write(f"Capture Rate is: {disc_best}\n")
                    con.write(f"The best A for {sla_nodes} nodes is {bestA}\n")
                    con.write(f"The best W is {bestW}\n")
                    con.write(f"The best n is {latency//(bestW+epoch_best/2)}\n")
                    con.write(f"The best nb is {bestNb}\n")
                con.write("\n")
            con.close() 