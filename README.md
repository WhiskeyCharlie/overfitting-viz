# ML-Overfitting-Plotly

An interactive application that demonstrates overfitting in machine learning models

## Getting Started

### Using the app

The **Dataset** dropdown lets you select among different datasets. Each dataset is identified by its degree; Degree 1 is a line, Degree 2 a parabola, etc.

**Noise Factor** lets you add noise to the sampled data. The value is proportional to the standard deviation of the (Gaussian) noise introduced to the data.

**Dataset Sample Size:** enables you to select the sample data size. Default is 30 data points, empirically the phenomenon of overfitting is easier to see with fewer points, somewhere in the neighborhood of 50 points should suffice.

**Model Polynomial Degree:** enables you to control the polynomial degrees to fit your data. E.g., selecting 1 fits a line, selecting 10 fits a degree 10 polynomial.

### Results

After each selection, you will see a graph representing the model with its MSE loss for the train and test sets.

Notice also the graph on the right-hand side. This graph shows how the MSE (mean-squared error) would change as you increased or decreased the degree of the polynomial fitting the data. The vertical green line marks the currently selected degree.

## Running the App From Source

The dependencies are listed in `requirements.txt`. The versions listed therein were tested against Python 3.10 on Ubuntu 22.04 (Jammy Jellyfish). While newer versions of these dependencies may also work, there is no guarantee of it.

While this guide should get you up and running, it is important to understand the basic usage of `virtualenv` and `pip` before proceeding.

### Virutalenv and Pip

#### One-time Installation

1. Create a new virtual environment. This is an isolated environment to store all the dependencies, so as not to pollute your system with our specific dependencies. We assume you are running this command in the terminal, but graphical options exist within certain IDEs such as PyCharm.

   Run: `virtualenv overfitting_venv` from the directory containing the downloaded files. (If you get an error to the effect of "virtualenv" not being found, you probably need to install it. Find a recent guide on installation for your operating system of choice) This command creates the isolated environment that we will install our dependencies in.

2. Activate the environment to begin using it. This step is platform specific. On Linux systems, the command should
   resemble
   `source overfitting_venv/bin/activate`. We recommend finding an up-to-date guide for your Operating System (Windows, Mac, etc.).

3. Install the requirements (Only required the first time you create the environment) with the
   command `pip install -r requirements.txt`

   This may take a while, as it downloads and installs numerous Python libraries.

#### Running the code (each time you want to launch the app).

1. Activate the virtualenv created at installation. This is the same procedure as in the Installation (Step 2).
2. From the same terminal in which you just activated the virtual environment, run `python app.py`. An IP address with a port will be printed, it could look like "http://127.0.0.1:2522/". Open this link in your web browser, and you should now see the app.
3. You can close the app by typing `ctrl+c` in the terminal running the code.

## Running the App as a Docker Image

### Prerequisites

Some basic knowledge of Docker will be useful for this section.  We assume you have installed Docker (this process differs from platform to platform).

This approach will make editing the code less convenient, so it is recommended mostly as a quick way to play around with the program. Good for demos, etc.

1. `docker pull ghcr.io/whiskeycharlie/overfitting-viz:master`
2. `docker run --rm -it -p 2522:2522 overfitting-viz:master`
