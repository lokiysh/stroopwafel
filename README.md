## STROOPWAFEL
Based on https://arxiv.org/abs/1905.00910

A short documentation of stroopwafel implementation can be found here -> https://docs.google.com/document/d/15To_ragEkT13gYitBoCdhq38Z97byrjGtS49Mq8cxG8

# Installation
Make sure you have python3 installed in the system, if not it can be downloaded from https://www.python.org/downloads/. 

To install this package simply run <code>pip3 install stroopwafel</code>. [ If you are installing on our slurm cluster helios, then <code>python3 -m pip install --user stroopwafel</code> should work if you dont have pip3 ]

This package has the following dependencies. They should be automatically installed for you when you install this package.
<ul>
    <li>numpy</li>
    <li>scipy</li>
</ul>

To test if its installed, go to python3. On the terminal: <code>python3</code>.
Inside the python prompt, <code>import stroopwafel</code> should not throw any errors.

# Running
Create a script similar to interface_genais.py in the test directory. Make sure you have the external application executable (such as COMPAS) defined in this script. Provide the other details and functions and run it using <code>python filename.py</code>. 

Note that stroopwafel is independent of the external application and does not have to reside in the same directory. 

Additionally you can pass in optional arguments in the command line:
<br/>
optional arguments:<br/>
  -h, --help            show this help message and exit<br/>
  --num_systems NUM_SYSTEMS
                        Total number of systems<br/>
  --num_cores NUM_CORES
                        Number of cores to run in parallel<br/>
  --num_per_core NUM_PER_CORE
                        Number of systems to generate in one core <br/>
  --debug DEBUG         If debug of COMPAS is to be printed <br/>
  --mc_only MC_ONLY     If run in MC simulation mode only <br/>
  --run_on_helios RUN_ON_HELIOS
                        If we are running on helios (or other slurm) nodes <br/>
  --output_filename OUTPUT_FILENAME
                        Output filename <br/>
</code>
