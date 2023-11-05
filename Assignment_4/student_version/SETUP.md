# CS 4464/7643 Deep Learning HW4 Local Environment Setup

To run the notebook locally, please do the following steps:

First, [install Anaconda](https://www.anaconda.com/) or any other conda installation you are comfortable with.

Second, [install VS Code](https://code.visualstudio.com/) along with the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

Then, open the **root homework folder** with VS Code.

Next, Run the following command below **once** in terminal: `conda env create -f ./environment.yml`

Finally, change the environment/ipykernel in the top of this VS Code window to `cs7643-a4`.

If you want to run anything within your command line, ensure that you have run `conda activate cs7643-a4` to change to the correct conda environment.