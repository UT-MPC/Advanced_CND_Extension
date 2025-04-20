#!/bin/bash

# Compile the Java program
javac BLEDiscSimulator.java

# Define parameters
params=("3" "10" "25" "50" "75" "100")
discProbList=(0.7 0.8 0.9)
latencyList=(10.0 30.0)
chunkList=(1)

# Loop over each parameter and run the simulation
for n_chunk in "${chunkList[@]}"; do
    for discProb in "${discProbList[@]}"; do
        for latency in "${latencyList[@]}"; do
            for param in "${params[@]}"; do
                java BLEDiscSimulator "./Extended/Extended_blend_${param}n_${discProb}_${latency}s.properties" \
                                    "./output/Extended_blend_${param}n_${discProb}_${latency}s.log" \
                                    "./output_dc/Extended_blend_${param}n_${discProb}_${latency}s.dc" \
                                    1000
            done
        done
    done
done
