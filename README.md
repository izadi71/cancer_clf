# Cancer Classification

This project uses machine learning to classify breast cancer tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) Dataset. The project aims to compare different classification models on the dataset to find the best performing model.

The project includes scripts for loading and preprocessing the data, selecting features, defining classifiers, training classifiers, evaluating classifiers, cross-validating classifiers, and visualizing evaluation metrics. The project also includes a `config.py` file that allows you to easily change the configuration variables for the project, such as file paths, hyperparameters, and other settings.


## Installation

To use this project, you will need to have Python 3 and `conda` installed on your computer. You can download Python 3 from the [official website](https://www.python.org/downloads/) and `conda` from the [official website](https://docs.conda.io/en/latest/miniconda.html).

Once you have installed Python 3 and `conda`, you can create an environment for this project by running the following command in the root directory of the project:

```sh
conda create --name cancer_clf python==3.9
```

This will create a new conda environment named cancer_clf with Python 3 installed. You can activate this environment by running the following command:

```sh
conda activate cancer_clf
```
Once you have activated the cancer_clf environment, you can install the necessary libraries for this project by running the following command in the root directory of the project:

```sh
conda install --file requirements.txt
```
This will install the libraries listed in the requirements.txt file using conda. Once you have installed these libraries, you can use this project as described in the Usage section.


## Usage

To use this project, you will need to set the `PYTHONPATH` environment variable to the root directory of the project. You can do this by running the following command in the root directory of the project:

```sh
export PYTHONPATH=${PWD}
```

Once you have set the `PYTHONPATH` environment variable, you can run the main script of the project by running the following command in the `src` directory of the project:

```sh
python src/main.py
```

This will run the main script of the project, which will load the data, split it into training and test sets, select features, define classifiers, train classifiers, evaluate classifiers, cross-validate classifiers, and visualize evaluation metrics.

You can modify the behavior of the script by changing the values of the configuration variables defined in the `config.py` file.



*** notice: tests are not complete ***


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.