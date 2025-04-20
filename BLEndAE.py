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


def compute_P_overlap_and_alpha(A, W):
    """
    Compute P_overlap for Case A and Case B and calculate alpha using the average P_overlap.

    Parameters:
        A (float): Length of the range.
        W (float): Slope parameter.

    Returns:
        dict: Contains P_overlap for Case A, Case B, average P_overlap, and alpha.
    """
    # Calculate P_overlap for Case A
    P_overlap_A = (A**3 / (3 * W**4) +
                   A * (-A**2 + 2 * W**2) / (2 * W**4) +
                   (A**4 - 4 * A**2 * W**2 + 4 * W**4) / (4 * A * W**4))
    
    # Calculate P_overlap for Case B
    P_overlap_B = (-A**3 / (3 * W**4) +
                   (A**4 - 4 * A**2 * W**2 + 4 * W**4) / (4 * A * W**4))
    
    # Calculate average P_overlap
    P_overlap_avg = (P_overlap_A + P_overlap_B) / 2

    # Calculate alpha as P_overlap_avg / (1 / A)
    alpha = P_overlap_avg / (1 / A)

    return {
        "P_overlap_A": P_overlap_A,
        "P_overlap_B": P_overlap_B,
        "P_overlap_avg": P_overlap_avg,
        "alpha": alpha
    }


def compute_n_probability(latency, window, n_target):
    """
    Calculate the probability that exactly n_target complete windows fit within the given latency
    and the expected remaining latency after placing these windows.

    Parameters:
        latency (float): The total available latency.
        window (float): The base size of a window (W).
        n_target (int): The target number of complete windows to fit.

    Returns:
        prob_n (float): Probability that exactly n_target windows fit within the latency.
        remainder_expectation (float): Expected remaining latency after n_target windows.
    """
    if n_target <= 0 or window <= 0:
        return 0, 0  # Invalid input; return zero probability and zero remainder.

    # Window size distribution parameters (Uniform[W, 2W])
    mean_W = 1.5 * window  # Mean of the window size.
    var_W = window**2 / 12  # Variance of the window size.
    std_W = np.sqrt(var_W)  # Standard deviation of the window size.

    # Sum of n_target windows (S_n) distribution parameters
    mean_S_n = n_target * mean_W  # Mean of S_n (sum of n_target windows).
    var_S_n = n_target * var_W  # Variance of S_n.
    std_S_n = np.sqrt(var_S_n)  # Standard deviation of S_n.

    # Determine valid range of S_n
    # lower_bound = max(n_target * window, 0, latency-window)  # Ensure enough space for n_target windows.
    lower_bound = max(n_target * window, 0, latency-mean_W+1)  # Ensure enough space for n_target windows.
    upper_bound = min(n_target * 2 * window, latency)  # Ensure sum of windows does not exceed latency.

    if lower_bound >= upper_bound:
        return 0, 0

    # Standardized bounds for S_n
    z_lower = (lower_bound - mean_S_n) / std_S_n  # Standardized lower bound.
    z_upper = (upper_bound - mean_S_n) / std_S_n  # Standardized upper bound.

    pdf_lower = norm.pdf(z_lower) / std_S_n
    pdf_upper = norm.pdf(z_upper) / std_S_n
    cdf_lower = norm.cdf(z_lower)
    cdf_upper = norm.cdf(z_upper)

    # Normalize p1 and p2
    total_prob = cdf_upper - cdf_lower
    if total_prob > 0:
        p1 = pdf_lower / total_prob
        p2 = pdf_upper / total_prob
    else:
        p1, p2 = 0, 0

    
    # Calculate probability of n_target windows fitting within the latency
    # Use the normal distribution CDF for the standardized range.
    prob_n_raw = cdf_upper - cdf_lower
    prob_n = prob_n_raw / (norm.cdf((latency - mean_S_n) / std_S_n) - norm.cdf((-mean_S_n) / std_S_n))  # Normalize.

    # Calculate the expected remaining latency (R = latency - S_n)
    # Integrate over the valid range of S_n
    def remainder_expectation_integrand(s_n):
        """
        Calculate the contribution to the expected remainder for a specific S_n value.

        Parameters:
            s_n (float): A potential sum of n_target windows.

        Returns:
            float: Contribution to the expected remainder at s_n.
        """
        remainder = latency - s_n  # Remaining latency after placing S_n.
        prob_density = norm.pdf((s_n - mean_S_n) / std_S_n) / std_S_n  # PDF of S_n (normalized).
        return remainder * prob_density

    s_n_values = np.linspace(lower_bound, upper_bound, 10)
    remainder_expectation = np.trapezoid(
        [remainder_expectation_integrand(s_n) for s_n in s_n_values], s_n_values
    )

    return prob_n, remainder_expectation



def compute_disc_prob(A, i, E, nb, C, psiGap, minN, W_base, latency):
    Window = W_base + W_base/2

    P_oneW = compute_disc_from_lat(A, nb, C, E, psiGap, lat=Window*(2/3), window=Window, W_base=W_base)  ###

    Pd_total = 0
    cnt = 0

    for start in range(0, min(int(Window), latency), 20):

        Pd = 0
        Pd_minus = 0
        Pd_plus = 0

        n = math.floor((latency-start)/Window)

        Pn, res = compute_n_probability(latency-start, W_base, n)
        Pn_minus, res_minus = compute_n_probability(latency-start, W_base, n-1)
        Pn_plus, res_plus = compute_n_probability(latency-start, W_base, n+1) 

        for i in range(minN, n+1):
            Pd += math.comb(n,i) * (P_oneW**i) * (1-P_oneW)**(n-i)
        for i in range(minN, n-1+1):
            Pd_minus += math.comb(n-1,i) * (P_oneW**i) * (1-P_oneW)**(n-1-i)
        for i in range(minN, n+1+1):
            Pd_plus += math.comb(n+1,i) * (P_oneW**i) * (1-P_oneW)**(n+1-i)


        lat_start = max(0, start-W_base/2) + (A+max_slop/2)/2
        P_start = compute_disc_from_lat(A, nb, C, E, psiGap, lat=max(0, lat_start), window=Window, W_base=W_base)

        lat_res = res//(A+max_slop/2) * (A+max_slop/2) if res >= L_Scan+B else 0
        P_res = compute_disc_from_lat(A, nb, C, E, psiGap, lat=min(lat_res, W_base), window=Window, W_base=W_base) ###

        if n>=minN-1>=0:
            Pd += math.comb(n,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n-minN+1) * (P_res+P_start - P_start*P_res)
        if n-1>=minN-1>=0:
            res = res_minus
            lat_res = res//(A+max_slop/2) * (A+max_slop/2) if res >= L_Scan+B else 0
            P_res = compute_disc_from_lat(A, nb, C, E, psiGap, lat=min(lat_res, W_base), window=Window, W_base=W_base) ###
            Pd_minus += math.comb(n-1,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n-1-minN+1) * (P_res+P_start - P_start*P_res)
        if n+1>=minN-1>=0:
            res = res_plus
            lat_res = res//(A+max_slop/2) * (A+max_slop/2) if res >= L_Scan+B else 0
            P_res = compute_disc_from_lat(A, nb, C, E, psiGap, lat=min(lat_res, W_base), window=Window, W_base=W_base) ###
            Pd_plus += math.comb(n+1,minN-1) * P_oneW**(minN-1) * (1-P_oneW)**(n+1-minN+1) * (P_res+P_start - P_start*P_res)

        Pd_total += (Pn*Pd + Pn_minus*Pd_minus + Pn_plus*Pd_plus) / (Pn+Pn_minus+Pn_plus)
        cnt += 1

    return Pd_total/cnt


def compute_disc_from_lat(A, nb, C, E, psiGap, lat=0, window=0, W_base=0):

    if lat<B: return 0

    k = math.floor(lat / E)
    leftover = lat - k * E
    fraction = (leftover / E)

    psi = W_base/2 

    extraBeaconRatio =  (max_slop/2) / (A + max_slop)
    gamma = (1-psi/window) * (C-2) * (1+extraBeaconRatio)

    Pnc = ( 1 - 2*B/(A+max_slop)) ** gamma

    P_in = (1-(psi+2*B)/window)
    P_in_frac = (1-(psi+2*B)/window)  * ((fraction*L_Scan-B)/(A+max_slop)) if fraction*L_Scan-B>0 else 0

    if k>=1:
        Pd = P_in*Pnc
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
                        
                        for W_base in range(int(latency/(2*minN+1)), int(4*L_Scan), -1): #A<E/4
                            
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
                    con.flush()
                con.write("\n")
            con.close() 