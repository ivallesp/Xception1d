# Xception-1-dimensional
[![Build Status](https://travis-ci.com/ivallesp/Xception1d.svg?branch=master)](https://travis-ci.com/ivallesp/Xception1d)
[![Code coverage](https://codecov.io/gh/ivallesp/Xception1d/branch/master/graph/badge.svg)](https://codecov.io/gh/ivallesp/xception1d)

Implementation of a neural network architecture for solving *speech recognition* tasks with *limited vocabulary* **out of the box**. This architecture is based on the Xception architecture, presented by *François Chollet* in 2017.
- It achieved state-of-the-art results with the *Google Tensorflow Speech Commands data set*, surpassing human performance in the most complex tasks.
- It works always in temporal domain, without needing to perform tedious and computationally expensive *Fourier transforms*
- It can be easily adapted to variable size audio clips and to different tasks

We suggest this architecture as the *de facto* solution when a voice commands recognition with restricted vocabulary task arises; considering the computing power is not a limiting factor.


# Getting started
If you are interested in run the code, please, follow the next steps.

1. Clone the repository
2. Navigate with your terminal inside the folder of the project and install the required libraries using the following command: `pip install -r requirements.txt`
3. Use the `settings_template.json` file in the root of the project as a template for creating a `settings.json` file and fill it with your configuration.
4. The directory config contains the settings for reproducing the results submitted with the paper. Choose one, select a seed and run it using the following command: `python main.py [config filepath] [seed]`

The seeds that have been used for generating the current results are the following ones `655321`, `655322`, `655323`, `655324`, `655325`.
Feel free to create new settings and store them in the config file to try new parameters.

# Contribution
If you wish to contribute in any way, please, submit a pull request
