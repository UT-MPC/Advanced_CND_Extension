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
b_sec = 2
AUX_offset = 3
B = (b + AUX_offset + b_sec)
L_Scan = A - B  ##Extended
I_secB = 7.98

sla_prob_list = [0.70, 0.80, 0.90]  # Desired probability of discovery
latency_list = [10000, 30000] 
sla_node_set = [3, 10, 25, 50, 75, 100]  # Set of nodes to test
n_chunk_list = [1]

def compute_Q(adv_interval, num_beacons, b, B, E):
    idletime = E - L_Scan - ((b * 3 + b_sec) * num_beacons)  # Idle time within the epoch
    return I_rx * L_Scan + num_beacons * (b * 3 * I_tx + b_sec * I_secB) + idletime * I_idle

# Function to compute the probability of discovery
def compute_disc_prob(A, E, nb, C):
    k = math.floor(latency / E)  # Number of full epochs that fit within the latency
    leftover = latency - k * E  # Remaining time not fitting into full epochs
    fraction = leftover / E  # Fractional epoch (should be less than 1)
    return compute_disc_prob_without_W(A, nb, C, k, leftover, fraction, E)

# Function to compute discovery probability without considering window overlap
def compute_disc_prob_without_W(A, nb, C, k, leftover, fraction, E):
    W = A  # Base window size
    P_in = (1 - (2 * B) / W)  # Probability of beacon being in range
    P_over = (2 * B) / W  # Probability of beacon overlapping
    gamma = (C - 2)  # Degree of contention among nodes
    Base = (2 * B)  # Base contention factor
    
    # Probabilities for primary and secondary channel collisions
    Pc_before = (W - 2.5 * B) / (W - 2 * B) * (B - b) / Base
    Pc_primary = (2 * b) / Base
    Pc_second = (b_sec - b) / (Base * 37)  # Accounting for 37 secondary channels
    Pc_collide = (Pc_before + Pc_primary + Pc_second)
    
    # Final probability of no collision
    Pnc = P_in * (1 - P_over * Pc_collide) ** gamma
    return Pnc

# Output file for results
for n_chunk in n_chunk_list:
    for sla_prob in sla_prob_list:
        for latency in latency_list:

            # Write results to output file
            out_file_name = f"/Users/hy.c/Desktop/NoD_Plus/BLEndAE/Extended_nihao_{sla_prob}_{latency/1000}s_{n_chunk}chunk_log.txt"

            with open(out_file_name, 'w') as con:

                # Iterate through different numbers of nodes
                for sla_nodes in sla_node_set:
                    disc_best = -1
                    epoch_best = 0
                    n_best = -1
                    A_best = -1
                    Q_min = float('inf')  # Minimum energy consumption
                    I_min = float('inf')  # Minimum instantaneous current
                    
                    maxA = latency
                    minA = int(max(3 * B + 2 * b + 1, 20))+1  # Minimum advertisement interval
                    
                    # Test various advertisement intervals and epoch sizes
                    for A in range(minA, maxA + 1):
                        n_max = math.floor(latency / A)
                        for n in range(1, n_max + 1):
                            E = A * n  # Total epoch duration
                            disc_prob = compute_disc_prob(A, E, n, sla_nodes)
                            
                            if disc_prob >= sla_prob:
                                Q = compute_Q(A, n, b, B, E)  # Compute energy consumption
                                I = Q / E  # Compute instantaneous current
                                if I < I_min:  # Check if this configuration is better
                                    I_min = I
                                    Q_min = Q
                                    epoch_best = E
                                    n_best = n
                                    A_best = A
                                    disc_best = disc_prob
                    
                    # Write the best results for this number of nodes
                    con.write(f"\nThe best values for {sla_nodes} nodes are: n={n_best}, A={A_best}\n")
                    con.write(f"Best discovery probability is: {disc_best}\n")