# ML-Overfitting-Plotly

An interactive application that demonstrates overfitting in machine learning models

## Getting Started

### Using the app

The **Dataset** dropdown lets you select among different datasets. Each dataset is identified by its degree; Degree 1 is
a line, Degree 2 a parabola, etc.

**Noise Factor** lets you add noise to the sampled data. The value is proportional to the standard deviation of the (
Gaussian) noise introduced to the data.

**Dataset Sample Size:** enables you to select the sample data size. Default is 300 data points, empirically the
phenomenon of overfitting is easier to see with fewer points, somewhere in the neighborhood of 50 points should suffice.

**Model Polynomial Degree:** enables you to control the polynomial degrees to fit your data. E.g., selecting 1 fits a
line, selecting 10 fits a degree 10 polynomial.

### Results

After each selection, you will see a graph representing the model with its MSE loss for the train and test sets.

Notice also the graph on the right-hand side. This graph shows how the MSE (mean-squared error) would change as you
increased or decreased the degree of the polynomial fitting the data. The vertical green line marks the currently
selected degree.

[comment]: <> (TODO: Add an option for virtualenv and pip, maybe distribute as an executable?)

## Running the App From Source

The dependencies are listed in `requirements.txt`. The versions listed therein were tested against Python 3.8 on Ubuntu
20.04 (Focal Fossa). While newer versions of these dependencies may also work, there is no guarantee of it.

While this guide should get you up and running, it is important to understand the basic usage of
`virtualenv` and `pip` before proceeding.

### Virutalenv and Pip

1. Create a new virtual environment. This is an isolated environment to store all the dependencies, so as not to pollute
   your system with our specific libraries. We assume you are running this command in the terminal, but graphical
   options exist within certain IDEs such as PyCharm. Run:

   `virtualenv overfitting_venv` from the directory containing the downloaded files.

2. Activate the environment to begin using it. This step is platform specific. On Linux systems, the command should
   resemble
   `source overfitting_venv/bin/activate`. We recommend finding an up-to-date guide for your Operating System (Windows, 
   Mac, etc.).

3. Install the requirements (Only required the first time you create the environment). Change directory (for example
   with the command `cd`) into `ML-Overfitting-Plotly-master` Run:
   
   `pip install -r requirements.txt`

This may take a while, as it downloads and installs numerous Python libraries.

4. Run the code.

In your terminal, run `python app.py`. An IP address with a port will be printed, it could look like
"http://127.0.0.1:2522/". Open this link in your web browser, and you should now see the app.

### Conda

This section requires writing, if someone would like to provide the appropriate instructions, please make a pull request
on this repository or email the author.

## Running the App as an Executable

1. Download the executable file to some known folder, for example `~/Downloads/`
2. Open a terminal and navigate to the folder mentioned in step 1.
3. Run the executable as you would any other, on unix systems you can run `./app` where "app"
   is replaced by the name of the executable you downloaded.
   You may need to change the permissions on the file to make it executable (e.g., `chmod +x app`).
   You should find a guide for doing this on your operating system.

## Running the App as a Docker Image
### Prerequisites
Some basic knowledge of Docker will be useful for this section.
We assume you have installed Docker (this process differs from platform to platform).

This approach will make editing the code less convenient, so it is recommended mostly as a quick way to play around
with the program. Good for demos, etc.

1. `docker pull TODO ADD IMAGE PATH`
2. `docker run --rm -p 2522:2522 TODO ADD IMAGE NAME`
