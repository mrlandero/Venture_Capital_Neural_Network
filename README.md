# Venture_Capital_Neural_Network
Predict whether funding applicants will be successful, we will create a binary classification model using a deep neural network.

---

## Technologies

This project leverages python 3.7 with the following packages:

**[Pandas Library](https://pandas.pydata.org/)** - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.<br>

**[Pathlib Library](https://pathlib.readthedocs.io/en/pep428/)** - This module offers a set of classes featuring all the common operations on paths in an easy, object-oriented way.<br>

**[TensorFlow Library](https://pypi.org/project/tensorflow/)** - TensorFlow is an open source software library for high performance numerical computation.<br>

**[TensorFlow Keras Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)** - Just your regular densely-connected NN layer.<br>

**[TensorFlow Keras Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)** - Sequential groups a linear stack of layers into a tf.keras.Model.<br>

**[SkLearn.model_selection train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)** - Split arrays or matrices into random train and test subsets.<br>

**[SkLearn.preprocessing Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)** - Standardize features by removing the mean and scaling to unit variance.<br>

**[SkLearn.preprocessing OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)** - Encode categorical features as a one-hot numeric array.

---

## Installation Guide

Before running the application first install the following dependencies:

You’ll need to install the following tools for this module:

1. TensorFlow 2.0 

2. Keras 

To do so, follow the examples in the next sections.

### Important

If you're using an Apple computer with an M1 Chip you will NOT install these tools. Please refer to the section below titled "Apple M1 Chip Users" for instructions on how to run the activities in this application.

## TensorFlow

The TensorFlow 2.0 library has several dependencies, which should already be installed in the default Conda environment. Please refer to the troubleshooting section below for details about this environment. Make sure to run the following commands with your Conda environment activated.

To install TensorFlow, open the terminal, and execute the following command:

Use the `pip install` command to install the TensorFlow 2.0 library.

```python
pip install --upgrade tensorflow
```

## Verify Installation

Once the TensorFlow install is complete, verify that the installation completed successfully.

```python
python -c "import tensorflow as tf;print(tf.__version__)"
```

The output of this command should show version 2.0.0 or higher.

## Keras

Keras is a popular deep learning framework that serves as a high-level API for TensorFlow. Keras is now included with TensorFlow 2.0. So, run the following command to verify that the package is available:

```python
python -c "import tensorflow as tf;print(tf.keras.__version__)"
```

The output should show version 2.2.4-tf or later.

## Troubleshooting

It can be frustrating when packages do not install correctly. Refer to the latest official TensorFlow Install Guide to troubleshoot. Alternatively, Google Colab works well with Tensorflow and can be used to run Jupyter Notebook files.

## Appple M1 Chip Users

The Apple M1 Chip is not currently compatible with a typical Tensorflow installation, but that's ok! You can easily run all of the sections in this application using Google Colab. To complete the sections in this application, visit Google Colab upload the notebook, and run the code as you would normally.

You are set with the installations now!


