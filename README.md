# Budgeted Machine Learning Framework for Bayesian Belief Networks

The Budgeted Learning with Bayesian Network Experiment Framework (BLBN-EF)
consists of two main components, in the form of computer programs: (1) the
_learner_ and (2) the _generator_.  Both of these computer programs are implemented
in the C programming language and make use of the BLBN application programming
interface (API), which is also implemented in the C programming language. The
BLBN API depends on the [Norsys Netica](https://www.norsys.com/netica.html) [C API](https://norsys.com/netica_c_api.htm) for its Bayesian network data
structures and algorithms.  The BLBN API does not include its own Bayesian
network data structure implementation.  The BLBN API provides a framework for
implementing budgeted machine learning algorithms that relies on the data
structures and algorithms provided by the Netica C API whenever possible.

The BLBN-EF and the BLBN API was built to be used in an academic research
environment.  It has been designed to be simple and extensible, so new
algorithms can be easily integrated into the API in a strightforward manner.
To effectively use the BLBN-EF and BLBN API, it is crucial to be familiar with
the Netica C API, since the BLBN-EF depends on the BLBN API, which in turn
depends on the Netica C API.

# How to Compile BLBN-EF Programs

The BLBN-EF must be compiled on the Linux operating system.  The BLBN-EF was
designed to be compatible with the Holland Computing Center's (HCC) Prairiefire
cluster, which is made up of nodes that run the Linux operating system.

To compile the learner and generator components of the BLBN-EF, run the
makefile using the `make` command in a Linux terminal (such as the Prairiefire
head node):

```
make
```

This will invoke the make command with the makefile named `Makefile`.  This
will produce two executables: (1) `blbn_learner` and (2) `blbn_generator`.  To
compile only the learner, run the command:

```
make learner
```

Likewise, to compile only the generator, run the command:

```
make generator
```

To inspect the commands that are executed when the makefile is invoked, open
the file named `Makefile` in a text editor such as Vim or Emacs.

Alternatively, the following command can be run to manually compile the
learner (however, this method should be considered a last resort, and should
be used only when the Makefile cannot be used):

```
/util/comp/gcc/4.4.1/bin/gcc ./lib/NeticaEx.o ./src/blbn/blbn.c \
	./src/blbn_learner.c -o blbn_learner -L"./lib" -lm \
	-lnetica -lpthread -lstdc++
```

Likewise, the following command can be run to manually compile the generator:

```
/util/comp/gcc/4.4.1/bin/gcc ./lib/NeticaEx.o ./src/blbn_generator.c \
	-o blbn_generator -I"./src" -L"./lib" -lm -lnetica \
	-lpthread -lstdc++
```

For instructions on running the compiled `blbn_learner` and `blbn_generator`
executables on Prairiefire, go to the section named _Running Experiments Using 
the BLBN-EF_.

For instructions on how to use the make command visit the [make documentation](http://www.gnu.org/software/make/manual/make.html).

For information about accessing the Holland Computing Center Prairiefire cluster or its successors, visit the [HCC website](http://hcc.unl.edu/).

# Running Experiments Using the BLBN-EF

To run an experiment using BLBN-EF, use one of the included run scripts.
Scripts are included for running the experiments on a single laptop or desktop
(running Linux) with processes that run in a serial or parallel manner, and for
submitting jobs to be run as independent processes on the HCC Prairiefire
cluster.  Because individual processes in experiments can take many hours or
days to complete, experiments should be run on the HCC Prairiefire cluster
whenever possible.

To run an experiment on the HCC Prairiefire cluster, run a Bash shell script
associated with the desired experiment.  For example, to run the experiment
using the ALARM data set and Bayesian network structure, run the following
command on the head node of the Prairiefire cluster:

```
submit_ALARM.sh
```

This Bash shell script contains parameters for the ALARM experiment, including
parameters specifying file locations of the data set and Bayesian network
structure, learning policy/algorithm, budget value, the directory to which
the results of the experiment should be saved, among others.  For a complete
list of the parameters, open the Bash script in a text editor.  An experiments
parameters are used by the script to submit jobs to the Prairiefire cluster
scheduler, which will schedule each job to run as an independent process on
one of the cluster's nodes.

A Bash shell script such as this must be created for each experiment.

## Creating and Configuring an Experiment

To create an experiment, a Bayesian network structure file is required.  This
file will have either the extension `*.dne` or `*.neta`.  Of these two extensions,
the extension `*.dne` should be preferred since it is a text file and can be
read and modified easily by humans in a text editor.  The extension `*.neta` is
a binary file format that is not intended to be viewed or modified in a text
editor.

## Archiving Experiment Results

Because an experiment produced many results files (i.e., output files) in the
`/results/` directory, it may be desirable to periodically compress and archive
all files produced by experiments.  The BLBN-EF has a script file that
produces a compressed archive (in tar.gz format) of the `/results/` directory and
all subdirectories that preserves the directory hierarchy and all file
permissions.  To execute this script, run the following command:

```
archive_results.sh
```

The produced archive will be named based on the current date
according to the format `results_YYYY-MM-DD.tar.gz` where YYYY is the current
year, MM is the current month, and DD is the current date.  For example, an
archive produced on June 30, 2011 would be named `results_2011-06-30.tar.gz`.

# Usage

## `blbn_generator`

Compile the generator with the following command:

```
make generator
```

Run the generator with the following command:

The short answer is that I generated the data sets using the `blbn_generator` program with input files that were the parameterized networks from the Netica Net Repository on the Norsys website. I wrote brief instructions about using the `blbn_generator` program in the README I wrote for you (which I've also attached to this email).  The general usage syntax for this program is as follows:

```
./blbn_generator -d <data_file_path> -m <model_file_path> -c <case_count> -t <target_node_name>
```

For example, to run the generator for the _BreastCancerWisconsin_ data set:

```
./blbn_generator -d ../data/BreastCancerWisconsin/BreastCancerWisconsin.cas -m ../data/BreastCancerWisconsin/BreastCancerWisconsin.dne -c 1000 -t Class
```

## Converting `*.neta` files to `*.dne` files

Here's the command that I used to generate a `*.dne` file from a `*.neta` file:

```
./blbn_generator -m ./data/Animals/Animals.neta
```

I have a file located at `./data/Animals/Animals.neta` and this command creates the file `./data/Animals/Animals.dne`.

# Applications

This code was used by Yaling Zheng in her doctoral research for her dissertation. Her code is available on GitHub in her [blbn_1st](https://github.com/YalingZheng/blbn_1st) and [blbn_2nd](https://github.com/YalingZheng/blbn_2nd) repositories. Her dissertation, _Machine Learning with Incomplete Information_ is available on [DigitalCommons@University of Nebraska - Lincoln](http://digitalcommons.unl.edu/dissertations/AAI3487272/).

The citation for Yaling's dissertation is below:

_Zheng, Yaling, "Machine learning with incomplete information" (2011). ETD collection for University of Nebraska - Lincoln. AAI3487272. http://digitalcommons.unl.edu/dissertations/AAI3487272_

# Dependencies

- [Norsys Netica](https://www.norsys.com/netica.html)
- [Netica APIs](https://www.norsys.com/netica_api.html)
- [Netica C API](https://www.norsys.com/netica_c_api.htm)
