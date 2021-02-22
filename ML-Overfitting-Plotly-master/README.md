# ML-Overfitting-Plotly
An interactive application that demonstrates overfitting in machine learning models

## Getting Started

### Using the app

The **Select Model** dropdown lets you select among different types of machine learning models. Currently only Linear Regression is implemented.

The **Select Dataset** dropdown lets you select among different datasets. To create a custom dataset, select the "Custom Data" option from the dropdown and go to Click Mode dropdown. 

The **Click Mode** dropdown lets you select among: add train data, add test data, or remove data points. Once an option is selected, click the graph to apply the chosen action.

**Select Dataset Noise Factor** lets you add noise to the sampled data. The value represents the standard deviation for the noise, which, by default, is sampled from a normal distribution with mean=0.

**Select Dataset Sample Size:** enables you to select the sample data size. Default is 300 data points.

**Select Model Polynomial Degree:** enables you to control the polynomial degrees to fit your data. Expects an integer value between 0-10 including.

After each selection, you will see a graph representing the model with its MSE loss for the train and test sets.

### Running the app locally

Create a virtual environment with conda and activate it.

```
conda create -n ml-overfitting-plotly python=3.6
conda activate ml-overfitting-plotly
```

Clone the git repo, then install the requirements with pip
```
git clone https://github.com/royee17/ML-Overfitting-Plotly.git
cd ML-Overfitting-Plotly
pip install -r requirements.txt
```

Run the app
```
python app.py
```