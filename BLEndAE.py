import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad
import collections
from scipy.optimize import fsolve
from functools import lru_cache


b = 3.3 / 3         # advertisement duration in ms, or minimum TX/RX overlapping, lambda in Ruth
max_slop = 10       # the maximum random slop added by BLE
I_tx = 2.64         # TX current draw, in mA
I_rx = 2.108        # RX current draw, in mA
I_aWP = 0.339       # warmup period from scan to advertising, in mA
I_sWP = 0.406       # warmup period from advertising to scan, in mA
I_idle = 0.339      # IDLE time between 2 beacons, in mA
battery = (225 * 3 / 2.1)  # battery capacity in mAh

A = 100

# Extended
b_sec = 0
AUX_offset = 0
B = b ### for legacy
L_Scan = A + max_slop + B 

sla_prob_list = [0.70, 0.80, 0.90]  # Desired probability of discovery
latency_list = [30000, 10000] # 10s has no results 
sla_node_set = [3, 10, 25, 50, 75, 100]  # Set of nodes to test


# Function to compute the energy cost of one epoch given this schedule
def compute_Q(adv_interval, num_beacons, psi, E):
    idletime = E - (adv_interval + B + max_slop) - (b * num_beacons) - psi
    return (I_rx * (adv_interval + B + max_slop) + num_beacons * b * I_tx + idletime * I_idle) / E


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

        nonL = (W_base-L_Scan)
        d = L_Scan if start-nonL>=L_Scan else max(0, start-nonL)
        move = min(d, start_comb)
        l = d-move
        L_start = (d+l)*move/2 / start_comb if start_comb else 0

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

        start_comb = max(0, min(W_base, l_start-start))
        P_valid_start = start_comb / (W_base**2 * 3/2)

        nonL = (W_base-L_Scan)
        d = L_Scan if start-nonL>=L_Scan else max(0, start-nonL)
        move = min(d, start_comb)
        l = d-move
        L_start = (d+l)*move/2 / start_comb if start_comb else 0
        
        def integrand_P(res):
            s = lat - res
            P_res = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= L_Scan else res ### DIFF FROM Nihao
            frac_single, _ = cal_frac(L_start, L_end, Pd_one)
            return pdf_irwin_hall(s, k, W_base) * P_res * frac_single
        
        # edge case
        if k==0:
            res = lat
            if res>2*W_base: 
                return 0
            P_valid_end = 1 if res <= W_base else (2*W_base - res)/W_base
            L_end = L_Scan if res >= L_Scan else res ### DIFF FROM Nihao
            frac_single, _ = cal_frac(L_start, L_end, Pd_one)
            return P_valid_start * P_valid_end * frac_single
        else:
            weighted_frac_single, _ = quad(integrand_P, 0, min(lat, 2*W_base))
            return P_valid_start * weighted_frac_single
    

    totalComb, _ = quad(integrand_totalComb, 0, l_start)
    integral_frac_single, _ = quad(integrand_frac_single, 0, l_start)

    frac_single = integral_frac_single / totalComb if totalComb else 0
    return totalComb, frac_single, 0


def func_Pd(Pd_one, minN, k, frac_single=0, frac_both=0): 
    Pd = 0
    for n_disc in range(minN, k+1):
        n_nd = k-n_disc
        P_dics = Pd_one**(n_disc)
        P_not_dics = (1-Pd_one)**n_nd
        Pd += math.comb(k, n_disc) * (P_dics) * (P_not_dics)
    
    # frac
    n_disc = minN-1
    if k>=n_disc>=0:
        P_dics = Pd_one**n_disc
        P_not_dics = (1-Pd_one)**(k-n_disc)
        Pd += math.comb(k, n_disc) * (P_dics) * (P_not_dics) * (frac_single*Pd_one - frac_both*Pd_one**2)
        
    # # frac
    # n_disc = minN-2
    # if k>=n_disc>=0:
    #     P_dics = Pd_one**n_disc
    #     P_not_dics = (1-Pd_one)**(k-n_disc)
    #     Pd += math.comb(k, n_disc) * (P_dics) * (P_not_dics) * (frac_both*Pd_one**2)

    return Pd


def compute_disc_prob(A, nb, C, minN, W_base, latency):
    
    window = W_base + W_base/2  #0~W_base
    k = math.floor(latency / window) 
    
    # only 2/3 function
    func = 2/3 
    non_func = 1/3
    
    P_in = func * (1-(2*B)/W_base)
    extraBeaconRatio =  (max_slop/2) / (A + max_slop)
    gamma = (C-2) * (1+extraBeaconRatio) 
    Pnc = (1-func * 2*B/(A+max_slop)) ** gamma
    Pd_one = P_in * Pnc   

    midK = math.floor(latency / window)
    maxK = latency // W_base
    minK = max(0, latency // (2*W_base)-1) # floor
    W = 0
    P = 0
    for i in range(minK-1, maxK+1): # minK-1 !
        window = 1.5*W_base
        gamma = (C-2) * (1+extraBeaconRatio) 
        P_in = func - (2*B)/window
        P_over = P_in * 2*B/(A+max_slop)
        Pnc = (1-P_over) ** gamma
        Pd_one = P_in * Pnc
            
        prob_k, frac_single, frac_both = combK(latency - W_base*i , i, W_base, Pd_one)
        P_k = func_Pd(Pd_one, minN, i, frac_single, frac_both) * prob_k
        P += P_k
        W += prob_k
    
    return P/W if W else 0 # sum W == 1


for n_chunk in n_chunk_list:
    for sla_prob in sla_prob_list:
        for latency in latency_list:
            
            # Write results to output file
            out_file_name = f"./[BLEndAE] Final Result/BLEndAE Results/AE_blend_{sla_prob}_{latency/1000}s_log.txt"
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
                    
                    minN = 9 
                    for W_base in range(int(latency/minN)//2, int(4*L_Scan), -1):
                        E = W_base

                        advTime = E - L_Scan
                        psiGap = advTime % (A + max_slop / 2) ## use for BLE determine last size
                        advCnt = math.floor(E/A) - 2
                        
                        disc_prob = compute_disc_prob(A, advCnt, sla_nodes, minN, W_base, latency)

                        if disc_prob >= sla_prob:
                            Q = compute_Q(A, advCnt, psiGap, E)
                            I = Q
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
                    con.write(f"The best n is {latency//(1.5*bestW)}\n")
                    con.write(f"The best nb is {bestNb}\n")
                    con.flush()
                con.write("\n")
            con.close() 