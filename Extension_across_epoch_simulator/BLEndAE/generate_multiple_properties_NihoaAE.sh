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

    while IFS= read -r line; do
        if echo "$line" | grep -q "The best epoch size for"; then
            # Extract the number of nodes and the T value from the line
            numNodes=$(echo "$line" | grep -oE "\d+" | head -1)
            T=$(echo "$line" | grep -oE "\d+" | tail -1)
            
            # Skip if T value is -1
            if [ "$T" == "0" ]; then
                continue
            fi
        elif echo "$line" | grep -q "The best A for"; then
            # Extract A value
            A=$(echo "$line" | grep -oE "\d+" | tail -1)
        elif echo "$line" | grep -q "The best W is"; then
            # Extract W value
            W=$(echo "$line" | grep -oE "\d+" | tail -1)
        elif echo "$line" | grep -q "The best n is"; then
            # Extract n value and handle -1 case
            n=$(echo "$line" | awk -F'is ' '{print $2}')

            # Set the filename using the extracted n value and the provided discProb value
            filename="./BLEndAE/AE_nihao_${numNodes}n_${discProb}_${latency}s.properties"

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

# T value for the simulation
T = $T

# length of a BLEnd listen (in ms, directly from the solver) [BLEnd, Extended]
# I *think* this is the adv_interval
L = $A

W = $W

# length of a beacon (in ms)
b = 1.1

# length of offloading on secondary channel (in ms) [Extended only]
b_second = 0

# length of AUX Offset * Offset Unit [Extended only]
AUX_Offset = 0

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

# the number of data chunks [only multi-adv]
chunks = 11


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
for discProb in "${discProbList[@]}"; do
    for latency in "${latencyList[@]}"; do
        generate_properties "../AE_nihao_${discProb}_${latency}s_log.txt" --disc_prob "${discProb}"
    done
done