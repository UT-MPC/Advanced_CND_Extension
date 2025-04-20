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

A = 100
n_slot = -1
N_window, n = -1, -1

# Extended
b_sec = 2
AUX_offset = 3
B = (b + AUX_offset + b_sec)
I_secB = 7.98
L_Scan = A - B  ##Extended Nihao

sla_prob_list = [0.70, 0.80, 0.90]  # Desired probability of discovery
latency_list = [10000, 30000] 
sla_node_set = [3, 10, 25, 50, 75, 100]  # Set of nodes to test

n_chunk_list = [1, 2, 3]

# Function to compute the energy cost of one epoch given this schedule
def compute_Q(adv_interval, num_beacons, psi, E):
    idletime = E - L_Scan - (b * num_beacons)
    return (I_rx * L_Scan + num_beacons * (b * I_tx) + idletime * I_idle) / E


def compute_n_probability(lat, window, n_target):
    if n_target<=0: 
        return 0
    if window==0: 
        return 0

    # Expected length of one segment
    mean_L = 1.5 * window
    var_L = window**2 / 12  # Variance of a uniform distribution U[window, 2*window]
    std_L = (var_L ** 0.5)  # Standard deviation of L

    # Total mean and standard deviation of the sum of n_target segments
    mean_total = n_target * mean_L
    std_total = (n_target * var_L) ** 0.5

    # Calculate boundaries for n_target segments
    lower_bound = lat - (n_target + 1) * mean_L
    upper_bound = lat - n_target * mean_L

    # Standardize bounds
    z_lower = lower_bound / std_total
    z_upper = upper_bound / std_total

    # Compute probability using normal CDF
    return norm.cdf(z_upper) - norm.cdf(z_lower)


def compute_disc_prob(A, i, E, nb, C, psiGap, minN, W_base, latency):
    global n 

    Window = W_base + W_base/2

    P_oneW = compute_disc_from_lat(A, nb, C, E, psiGap, lat=Window*(2/3), window=Window, W_base=W_base)  ###

    Pd_total = 0
    cnt = 0

    for start in range(min(int(Window), latency)):

        Pd = 0
        Pd_minus = 0
        Pd_plus = 0

        n = math.floor((latency-start)/Window)

        Pn = compute_n_probability(latency-start, W_base, n)
        Pn_minus = compute_n_probability(latency-start, W_base, n-1)
        Pn_plus = compute_n_probability(latency-start, W_base, n+1)

        res = (latency - start - Window*n) ###

        for i in range(minN, n+1):
            Pd += math.comb(n,i) * (P_oneW**i) * (1-P_oneW)**(n-i)
        for i in range(minN, n-1+1):
            Pd_minus += math.comb(n-1,i) * (P_oneW**i) * (1-P_oneW)**(n-1-i)
        for i in range(minN, n+1+1):
            Pd_plus += math.comb(n+1,i) * (P_oneW**i) * (1-P_oneW)**(n+1-i)

        lat_start = max(0, start-W_base/2)
        lat_start = (lat_start//A) * A
        P_start = compute_disc_from_lat(A, nb, C, E, psiGap, lat=max(0, lat_start), window=Window, W_base=W_base)

        lat_res = (res//A + 1) * A
        P_res = compute_disc_from_lat(A, nb, C, E, psiGap, lat=min(lat_res, W_base), window=Window, W_base=W_base) ###

        if n>=minN-1>=0:
            Pd += math.comb(n,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n-minN+1) * (P_res+P_start - P_start*P_res)
        if n-1>=minN-1>=0:
            Pd_minus += math.comb(n-1,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n-1-minN+1) * (P_res+P_start - P_start*P_res)
        if n+1>=minN-1>=0:
            Pd_plus += math.comb(n+1,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n+1-minN+1) * (P_res+P_start - P_start*P_res)

        Pd_total += (Pn*Pd + Pn_minus*Pd_minus + Pn_plus*Pd_plus) / (Pn+Pn_minus+Pn_plus)

        cnt += 1

    return Pd_total/cnt

def compute_disc_from_lat(A, nb, C, E, psiGap, lat=0, window=0, W_base=0):

    if lat<B:
        return 0
    
    k = math.floor(lat / E)
    leftover = lat - k * E
    fraction = (leftover / E)

    psi = W_base/2

    # Extended
    Base = (2 * B)
    Pc_before = ((L_Scan-B) - B/2) / (L_Scan-B) * (B - b) / Base ### different from BLEnd
    Pc_primary = (2 * b) / Base
    Pc_second = max(0, b_sec - b) / Base / 37  # there are 37 secondary channels
    Pc_internal = (Pc_before + Pc_primary + Pc_second)
    
    P_over = 2*B/A 
    P_over = P_over*Pc_internal
    gamma = (1-psi/window) * (C-2)
    
    Pnc = (1-P_over) ** gamma

    if k>=1:
        P_in = (1-psi/window) * (1-(2*B)/A)
        Pd = P_in * Pnc
    else:
        P_in  = max(0, (1-psi/window) * (fraction*L_Scan-B)/A)
        Pd = P_in * Pnc

    return Pd


for n_chunk in n_chunk_list:
    for sla_prob in sla_prob_list:
        for latency in latency_list:
            
            # Write results to output file
            out_file_name = f"/Users/hy.c/Desktop/NoD_Plus/BLEndAE/ExtendedAE_nihao_{sla_prob}_{latency/1000}s_{n_chunk}chunk_log.txt"

            with open(out_file_name, 'w') as con:

                hrs = []  # Expected battery life in hours
                for sla_nodes in sla_node_set:
                    disc_best = -1
                    epoch_best = -1
                    bestA = -1
                    bestK = -1
                    bestNb = -1
                    bestW = -1
                    bestN = -1 

                    Q_min = np.iinfo(np.int32).max
                    I_min = np.iinfo(np.int32).max

                    for A in range(100, 101):

                        L_Scan = A - B 
                        minN = math.floor(0.8 * n_chunk) + (1 if n_chunk * 0.8 % 1 > 0 else 0)  # minimum number of windows that needs to be captured
                        
                        for W_base in range(int(latency / (2 * minN + 1)), int(4 * A), -1): # make sure there are enough complete windows
                            
                            if W_base%A!=0: continue

                            n_slot = W_base//A 
                            E = n_slot*A

                            if W_base < E:
                                break

                            advTime = E - L_Scan
                            psiGap = advTime % (A + max_slop / 2)  # use for BLE to determine last size
                            advCnt = n_slot
                            
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
                                    bestN = n_slot
                                    N_window = n

                    # Write results to the file
                    con.write(f"\nThe best epoch size for {sla_nodes} nodes is {epoch_best}\n")
                    con.write(f"Capture Rate is: {disc_best}\n")
                    con.write(f"The best A for {sla_nodes} nodes is {bestA}\n")
                    con.write(f"The best W is {bestW}\n")
                    con.write(f"The best n is {bestN}\n")
                    con.write(f"The best nb is {bestNb}\n")
                    con.write(f"The best N_window is {N_window}\n")
                con.write("\n")
            con.close()