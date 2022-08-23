Here is all the python code.


See the pysindy package: https://github.com/dynamicslab/pysindy
- look at the readme file under the heading 'Installation' to install pysindy before trying to run the code

All the SINDy, SINDy-PI and Ensemble-SINDy peices of code are based on examples from pysindy, see: https://github.com/dynamicslab/pysindy/tree/master/examples

For details on how the code works/how to call the functions used, see: https://github.com/dynamicslab/pysindy/tree/master/pysindy


For abrupt-SINDy, see: https://github.com/Ohjeah/abrupt-sindy

## Python files - descriptions 

### Testing pysindy functions

all_libraries_examples.py
-	Example of each method to build a library of candidate functions, for use with pysindy

### Lotka Volterra

lotkavolterra.py
-	Produces a time and a phase plot of the lotka-volterra equations 
-	Defines the function for the equations that’s called in ‘lotka_volterra_sindy.py’
lotka_volterra_sindy.py
-	Generates synthetic data from the Lotka Volterra system, and applies SINDy

### Synthetic data

second_law.py
-	Plots solutions to 2nd law equations w/ different forcing terms – constant, spring, damper, periodic term, and combinations of these
spring_damper_sin_noise.py
-	Plots solutions of the set of 1st order ODEs that represents the 2nd order ODE F = ma, where F is a has a spring term, a damper term, and a sinusoidal term 
-	Plots displacement, velocity, and accelerations
-	Solves the ODEs and adds noise 

### Using SINDy w/ the synthetic data

spring_damp_sin_sindy.py 
-	Generates synthetic data from the spring mass damper & sin term system, and applies SINDy
spring_damp_sin_sindypi.py
-	Generates synthetic data from the spring mass damper & sin term system, and applies SINDy-PI
spring_damp_sin_ens-sindy.py
-	Generates synthetic data from the spring mass damper & sin term system, and applies Ensemble-SINDy

### Applying SINDy to datasets

#### Case western 
*Note for using these files you’ll have to change the file path for the case western .mat datasets*

normal_baseline.py
-	Time and frequency plot of the defined Case Western datasets – normal operation, 0 hp, drive end and fan end sensor data 

12k_drive_fault.py
-	Time and frequency plot of the defined Case Western dataset – (Inner race fault: 7 mils, 0 hp, 12k samp/sec) Drive End Operation

12k_fan_fault.py
-	Time and frequency plot of the defined Case Western dataset – (Ball fault: 7 mils, 0 hp, 12k samp/sec) Fan End Operation

normal_sindy.py
-	Applied SINDy to either normal operation or normal w/ faulty appended (can comment out lines as necessary to choose what files to use) 

normal_sindypi.py
-	Applied SINDy-PI to either normal operation or normal w/ faulty appended (can comment out lines as necessary to choose what files to use) 
-	Note: this one is a little slow

normal_in_ens_sindy.py 
-	Applies ensemble-SINDy to either normal operation or normal w/ faulty appended (can comment out lines as necessary to choose what files to use) 
-	V1, V2, V3, and V4 

7mil_0hp_inner_fault_sindy.py
-	Applies SINDy, SINDy-PI, and ensemble-SINDy to the defined Case Western dataset – (Inner race fault: 7 mils, 0 hp, 12k samp/sec) Drive End Operation

#### NASA bearing 
nasa_bearing_example.py
-	Can choose to import the dataset of either of the 3 tests for the NASA Bearng data sets
-	Plots the chosen dataset – each channel against time, as well as RMS values, kurtosis values, shape, impulse, crest, and entropy 
-	Makes .csv files for each channel (which can then be used for ‘nasa_bearing_sindy.py‘
-	Note: this takes a while to run 
nasa_bearing_sindy.py
-	Applies SINDy, SINDy-PI, and ensemble-SINDy to the defined NASA Bearing dataset (from the .csv files created in ‘nasa_bearing_example.py’

