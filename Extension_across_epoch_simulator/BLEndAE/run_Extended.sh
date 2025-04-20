# #!/bin/bash

# # Compile the Java program
# javac BLEDiscSimulator.java

# # Define parameters
# params=("3" "10" "25" "50" "75" "100")
# discProbList=(0.7 0.8 0.9)
# latencyList=(10.0 30.0)
# chunkList=(1)

# ## TEST 
# params=("")
# discProbList=(0.9)
# # discProbList=(0.7)
# # discProbList=(0.8)
# latencyList=(30.0)

# # Loop over each parameter and run the simulation
# for n_chunk in "${chunkList[@]}"; do
#     for discProb in "${discProbList[@]}"; do
#         for latency in "${latencyList[@]}"; do
#             for param in "${params[@]}"; do
#                 java BLEDiscSimulator "./BLEndAE/Extended_${param}n_${discProb}_${latency}s_${n_chunk}chunk_blend.properties" \
#                                     "./output/Extended_${param}n_${discProb}_${latency}s_${n_chunk}chunk_blend.log" \
#                                     "./output_dc/Extended_${param}n_${discProb}_${latency}s_${n_chunk}chunk_blend.dc" \
#                                     50
#             done
#         done
#     done
# done
