# AI Poincare 2.0 

## File Structure 
1. saved_checkpoints/ -> contains all outputs, i.e. fractal method plots, loss plots, hamiltonian smears, etc. The outputs of each different system are in a folder with the name of the system (e.g. one_d_harmonic_damped) for the Damped 1D Harmonic Oscillator. 
2. configs/ -> contains the config.txt files, i.e. the parameters for each system.
3. oracle_data/ -> contains the oracle data, i.e. the z for each system (f_z for each system is created (according to the system dynamics) in utils.py).

## Code Flow 
1. poincare_2.py is the main file that does the following 
    a. Load a config file (which system to take is passed as a parameter "-config"),
    b. Set the seed according to the config (OR passed via the "--seed" parameter)
    c. Initialize a neural network using the config, from poincare_net.py (poincare_net.py contains the NN class)
    d. Load the data (using load_dataset from utils.py. f_z for each system is created (according to the system differential equations) in utils.py)
    e. Train the network (train function defined in nn_utils.py), save the results and the model weights.
2. run_experiments.sh -> trains an NN using different seeds and saves the model weights
3. fractal_method.py -> Loads the model weights of the above experiments and generates the fractal method curves/hamiltonian smears etc. 

## Creating a new dynamic system
In nutshell, to run an experiment on a new system we must do the following -
1. create a config file 
2. in utils.py, define the differential equations to create f_z from z
3. run poincare_2.py, passing the created config file as a parameter. This will train the model for 1 random seed
4. to train the model for many random seeds, we go to run_experiments.sh (changing the config file parameter inside run_experiment.sh)
5. run fractal_method.py to generate the dimensionality plots, hamiltonian smears, etc. (basically all plots that required us to use many networks trained using different seeds. this is why this step is in a separate file and not integrated with poincare_2.py itself).
6. the results will now appear in saved_checkpoints/<system name as defined in the config file>/ 