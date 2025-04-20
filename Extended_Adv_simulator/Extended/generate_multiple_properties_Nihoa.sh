#!/bin/bash

# Function to generate properties
generate_properties() {
    discProb=""

    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --disc_prob) discProb="$2"; shift ;;
            *) PARAMS_FILE="$1" ;;
        esac
        shift
    done

    # Check if disc_prob is provided
    if [ -z "$discProb" ]; then
        echo "Error: --disc_prob argument is required."
        echo "Usage: $0 --disc_prob <disc_prob_value> <path_to_params_file>"
        return 1
    fi

    # Read each line from the PARAMS_FILE
    while IFS= read -r line; do
        if echo "$line" | grep -q "The best values for"; then
            # Extract the n and A values from the line using sed
            numNodes=$(echo "$line" | grep -oE "\d+" | head -1)
            n=$(echo "$line" | awk -F'n=' '{print $2}' | awk -F',' '{gsub(/[[:space:]]+/, "", $1); print $1}')
            A=$(echo "$line" | awk -F'A=' '{print $2}' | awk '{gsub(/[[:space:]]+/, "", $1); print $1}')
            T=$(echo "$A * $n" | bc)

            # Check if n and A were extracted successfully
            if [ -z "$n" ] || [ -z "$A" ]; then
                echo "Error: Failed to extract n or A values from line: $line"
                continue
            fi

            # Set the filename using the extracted n value and the provided discProb value
            filename="/Users/hy.c/Desktop/NoD_Plus/Extended/Extended_Adv_simulatior/Extended/Extended_nihao_${numNodes}n_${discProb}_${latency}s.properties"

            # Create the properties file with the extracted values
            cat > "$filename" <<-EOF

# properties file for BLE Discovery Simulation Parameters

# protocol (options: blend, searchlight, nihao)
protocol = nihao

# log style (options: brief, verbose, or cdf)
logStyle = cdf

# model collisions (true or false)
modelCollisions = true

# model BLE's three advertising channels (true or false)
modelChannels = false

# number of devices in the simulation
numNodes = $numNodes

# T value for the simulation (set to A as extracted)
T = $T

# length of a BLEnd listen (in ms, directly from the solver) [BLEnd, Extended]
# I *think* this is the adv_interval
L = 100

# length of a beacon (in ms)
b = 1.1

# length of offloading on secondary channel (in ms) [Extended only]
b_second = 2

# length of AUX Offset * Offset Unit [Extended only]
AUX_Offset = 3

# pack loss rate
m = 0

# scan warm up interval
wp_scan = 0
wp_adv = 0

# Do we revise the model and add extra length(wp=wp_scan+wp_adv) into actual listening interval?
addExtra = false

# slot length (in ms) [not used in BLEnd & Extended]
slotLength = 55

# n (width and height of Nihao matrix) [only Nihao]
n = $n

# simulation time in milliseconds
simulationTime = 100000

# max random advertisement delay (BLE adds [0-10ms] random delay between advertising events)
maxAdditionalAdvDelay = 10

# mechanism for providing discovery guarantees, even when BLE adds a random delay to the advInterval [BLEnd]
# value should be NONE, LISTEN, or ADVERTISE
correctAdvDelayType = LISTEN

# whether bidirectional discovery (with extra opportunistic beacons) is enabled [BLEnd]
bidirectionalDiscoveryEnabled = true

# whether to save the entire simulation schedule
saveSimulationSchedule = false

# the file in which to save schedules
scheduleSaveFile = mySavedSchedule

# whether to load the entire simulation schedule from a file instead of generating one
loadSimulationSchedule = false

# the file from which to load schedules
scheduleLoadFile = mySavedSchedule

# show the schedule visualization?
showSchedules = false

# print statistics when done?
printStatistics = true
EOF
            echo "Properties file generated: ${filename}"
        fi
    done < "$PARAMS_FILE"
}

# Main execution starts here
params=("3" "10" "25" "50" "75" "100")
latencyList=(10.0 30.0)
discProbList=(0.7 0.8 0.9)
chunkList=(1)
for n_chunk in "${chunkList[@]}"; do
    for discProb in "${discProbList[@]}"; do
        for latency in "${latencyList[@]}"; do
            generate_properties "/Users/hy.c/Desktop/NoD_Plus/BlendAE/Extended_nihao_${discProb}_${latency}s_${n_chunk}chunk_log.txt" --disc_prob "${discProb}"
        done
    done
done