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
latency_list = [30000, 10000] 
sla_node_set = [3, 10, 25, 50, 75, 100]  # Set of nodes to test

n_chunk_list = [1, 2, 3]

# Function to compute the energy cost of one epoch given this schedule
def compute_Q(adv_interval, num_beacons, E):
    idletime = E - L_Scan - (b * num_beacons)
    return (I_rx * L_Scan + num_beacons * (b * I_tx) + idletime * I_idle) / E

def cal_frac(L1, L2, Pd_one):
    frac_start, frac_end = (L1-B)/(L_Scan-B), (L2-B)/(L_Scan-B)
    frac_start, frac_end = max(frac_start, 0), max(frac_end, 0)
    return frac_start+frac_end, frac_start*frac_end # single, both

def pdf_irwin_hall(x, k, W):
    if x < 0 or x > k*W:
        return 0.0
    m = math.floor(x/W)
    acc = 0.0
    for j in range(m + 1):
        acc += (-1)**j * math.comb(k, j) * (x - j*W)**(k-1)
    return acc / (math.factorial(k-1) * W**k)

def combK(latency: int, k: int, W_base: int, Pd_one) -> float:
    if latency==0:
        return 0, 0, 0

    l_start = 2*W_base

    def integrand_totalComb(start):
        lat = max(0, latency - start)
        
        start_comb = min(W_base, l_start-start)
        P_valid_start = start_comb / (W_base**2 * 3/2)

        nonL = (W_base-A)
        L_max = L_Scan if start-nonL>=L_Scan else max(0, start-nonL) ###
        diff = min(L_max, max(0,start_comb-B))
        L_min = L_max-diff
        L_start = (L_max+L_min)*diff/2 / start_comb if start_comb else 0

        def integrand_P(res):
            s = lat - res
            P_res = 1 if res <= W_base else (2*W_base - res)/W_base
            return pdf_irwin_hall(s, k, W_base) * P_res

        # edge case
        if k==0:
            res = lat
            if res>2*W_base: 
                return 0
            P_valid_end = 1 if res <= W_base else (2*W_base - res)/W_base
            return P_valid_start * P_valid_end
        
        P_valid_end, _ = quad(integrand_P, 0, min(lat, 2*W_base))
        return P_valid_start * P_valid_end

    def integrand_frac_single(start):
        lat = max(0, latency - start)

        start_comb = min(W_base, l_start-start)
        P_valid_start = start_comb / (W_base**2 * 3/2)

        nonL = (W_base-A)
        L_max = L_Scan if start-nonL>=L_Scan else max(0, start-nonL)
        diff = min(L_max, max(0,start_comb-B)) 
        L_min = L_max-diff
        L_start = (L_max+L_min)*diff/2 / start_comb if start_comb else 0
        
        def integrand_P(res):
            s = lat - res
            P_res = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= A else max(0, res-B) ### DIFF FROM BLEND
            frac_single, _ = cal_frac(L_start, L_end, Pd_one)
            return pdf_irwin_hall(s, k, W_base) * P_res * frac_single
        
        # edge case
        if k==0:
            res = lat
            if res>2*W_base: 
                return 0
            P_valid_end = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= A else max(0, res-B) ### DIFF FROM BLEND
            frac_single, _ = cal_frac(L_start, L_end, Pd_one)
            return P_valid_start * P_valid_end * frac_single
        else:
            weighted_frac_single, _ = quad(integrand_P, 0, min(lat, 2*W_base))
            return P_valid_start * weighted_frac_single
    
    def integrand_frac_both(start):
        lat = max(0, latency - start)

        start_comb = min(W_base, l_start-start)
        P_valid_start = start_comb / (W_base**2 * 3/2)

        nonL = (W_base-A)
        L_max = L_Scan if start-nonL>=L_Scan else max(0, start-nonL)
        diff = min(L_max, max(0,start_comb-B)) 
        L_min = L_max-diff
        L_start = (L_max+L_min)*diff/2 / start_comb if start_comb else 0
        
        def integrand_P(res):
            s = lat - res
            P_res = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= A else max(0, res-B) ### DIFF FROM BLEND
            _, frac_both = cal_frac(L_start, L_end, Pd_one)
            return pdf_irwin_hall(s, k, W_base) * P_res * frac_both

        # edge case
        if k==0:
            res = lat
            if res>2*W_base: 
                return 0
            P_valid_end = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= A else max(0, res-B) ### DIFF FROM BLEND
            _, frac_both = cal_frac(L_start, L_end, Pd_one)
            return P_valid_start * P_valid_end * frac_both
        else:
            weighted_frac_both, _ = quad(integrand_P, 0, min(lat, 2*W_base))
            return P_valid_start * weighted_frac_both
    

    totalComb, _ = quad(integrand_totalComb, 0, l_start)
    integral_frac_single, _ = quad(integrand_frac_single, 0, l_start)
    integral_frac_both, _ = quad(integrand_frac_both, 0, l_start)

    frac_single = integral_frac_single / totalComb if totalComb else 0
    frac_both = integral_frac_both / totalComb if totalComb else 0
    return totalComb, frac_single, frac_both

# when there is k complete windows, what is the capture probability
def func_Pd(Pd_one, minN, k, frac_single=0, frac_both=0): 
    if k<minN-1:
        return 0
    
    Pd = 0
    for n_disc in range(minN, k+1):
        n_nd = k-n_disc # number of window that is not discovered
        P_dics = Pd_one**(n_disc)
        P_not_dics = (1-Pd_one)**n_nd
        Pd += math.comb(k, n_disc) * (P_dics) * (P_not_dics)
    
    # frac
    if minN-1>=0:
        n_disc = minN-1
        P_dics = Pd_one**n_disc
        P_not_dics = (1-Pd_one)**(k-n_disc)
        Pd += math.comb(k, n_disc) * (P_dics) * (P_not_dics) * (frac_single-frac_both)*Pd_one

    return Pd

def compute_disc_prob(A, nb, C, minN, W_base, latency):
    window = W_base + W_base/2  #0~W_base
    
    # only 2/3 function
    func = 2/3 
    non_func = 1/3

    # Extended
    Base = (2 * B)
    Pc_before = ((L_Scan-B) - B/2) / (L_Scan-B) * (B - b) / Base 
    Pc_primary = (2 * b) / Base
    Pc_second = max(0, b_sec - b) / Base / 37  # there are 37 secondary channels
    Pc_internal = (Pc_before + Pc_primary + Pc_second)
    
    P_over = func * 2*B/A
    P_c = P_over*Pc_internal
    
    # gamma = func * (C-2)
    P_in = (1 - non_func) * (1-2*B/A)
    gamma = (C-2)
    
    Pnc = (1-P_c) ** gamma
    Pd_one = P_in * Pnc

    maxK = latency // W_base
    minK = max(0, latency//(2*W_base) -1) #  minK-1 !
    W = 0
    P = 0
    for i in range(minK, maxK+1):
        prob_k, frac_single, frac_both = combK(latency - W_base*i , i, W_base, Pd_one)
        P_k = func_Pd(Pd_one, minN, i, frac_single, frac_both) 
        P += P_k * prob_k
        W += prob_k
        
    return P/W if W else 0 ## sumW==1


for n_chunk in n_chunk_list:
    for sla_prob in sla_prob_list:
        for latency in latency_list:
            
            # Write results to output file
            out_file_name = f"./[Nihao Extended+AE] Final Result/Nihao ExtendedAE Results/ExtendedAE_nihao_{sla_prob}_{latency/1000}s_{n_chunk}chunk_log.txt"

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

                    minN = n_chunk
                    
                    max_n_slot = int(latency / A)
                    for n_slot in range(max_n_slot, 0, -1):
                        W_base = n_slot * A
                        advCnt = n_slot
                        
                        disc_prob = compute_disc_prob(A, advCnt, sla_nodes, minN, W_base, latency)

                        if disc_prob >= sla_prob:
                            Q = compute_Q(A, advCnt, W_base)
                            I = Q
                            if I < I_min:
                                I_min = I
                                Q_min = Q
                                epoch_best = W_base
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