# Advanced_CND_Extension

This simulator supports CND (Continuous Neighbor Discovery) extensions from the paper "Advanced Continuous Neighbor Discovery Methods for Enhancing Data Exchange". The repository contains two simulation components: (1) `Extension_across_epoch_simulation`, which includes extensions such as CodingAW and ExtAW built on top of BLEnd and Nihao protocols; and (2) `Extended_Adv_simulation`, which contains the ExtBeacon variant implemented separately. 

The analytical models used to generate simulation input parameters are implemented in Python located in the root directory, including BLEndAE.py, BLEndExtended.py, ExtendedAE.py, NihaoAE.py, NihaoExtended.py, and NihaoExtendedAE.py.

To run simulations, first use one of the `generate_multiple_properties_*.sh` scripts to generate the necessary `.properties` configuration files. These scripts use the output from the Python analytical models to define discovery probability (disc_prob), number of nodes, latency, and other protocol-specific settings. For example, you can run `./generate_multiple_properties_ExtAW.sh`, `./generate_multiple_properties_CodingAW.sh`, or `./generate_multiple_properties_Nihoa_ExtendedAW.sh`. You may edit the scripts to adjust parameter ranges or values as needed.

After the `.properties` files are generated, execute the corresponding `run_*.sh` script to perform simulations. These scripts will automatically detect the generated `.properties` files and invoke the simulator multiple times using appropriate settings. For example, use `./run_Extended.sh`, `./run_BLEndAW.sh`, or `./run_NihaoExtendedAW.sh`. All Java simulation logic is handled internally via the shell scripts, so no manual compilation is required.

Simulation results are saved into `.log` files, named according to the protocol and configuration. The output format is determined by the `.properties` settings—`printstatistics = true` enables output of average duty cycle per node, while `format = cdf` prepares data suitable for CDF plotting.

All steps are driven via the `.sh` scripts. To run the full pipeline: (1) execute the analytical Python script to compute optimal parameters; (2) run the appropriate `generate_*.sh` script to create properties files; (3) run the corresponding `run_*.sh` script; and (4) analyze the resulting `.log` files for evaluation or visualization.
