.. raw:: html

    <h1 align="center">
    <br>
    <a href="http://duet.openmined.org/"><img src="https://github.com/OpenMined/design-assets/raw/master/logos/OM/mark-primary-trans.png" alt="SyMPC" width="200"></a>
    <br>
    SyMPC
    </h1>
    <h4 align="center">
    A library that extends PySyft with SMPC support
    <br>
    <br>
    </h4>

    <a href=""><img src="https://github.com/OpenMined/SyMPC/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://openmined.slack.com/messages/support"><img src="https://img.shields.io/badge/chat-on%20slack-7A5979.svg" /></a>
    <a href="https://codecov.io/gh/OpenMined/SyMPC"><img src="https://codecov.io/gh/OpenMined/SyMPC/branch/main/graph/badge.svg?token=TS2rZyJRlo" /></a>


SyMPC **/ˈsɪmpəθi/** is a library which extends `PySyft <https://github.com/OpenMined/PySyft>`_ ≥0.3 with SMPC support. It allows computing over encrypted data, and to train and evaluate neural networks.


Installation
############

SyMPC is a companion library for PySyft. Therefore, we will need to install PySyft among other dependencies. We recommend using a virtual environment like `conda`.

.. code:: bash

    $ conda create -n sympc python=3.9
    $ conda activate sympc
    $ pip install -r requirements.txt
    $ pip install .

You can also run SyMPC using docker by running the following commands. 

.. code:: bash
    
    $ docker build -t sympc -f docker-images/Dockerfile .  #builds image named sympc
    $ docker run -i -t sympc  #runs the container 


Getting Started
###############

If we want to start learning how to use SyMPC we can go to the *examples* folder and execute the *introduction.ipynb*.

.. code:: bash
    
    $ conda activate sympc
    $ pip install jupyter
    $ jupyter notebook examples/introduction.ipynb

If we decided to use docker, we would need to run the image and publish the jupyter notebook port

.. code:: bash
    
    $ sudo docker run -i -t -p 8888:8888 sympc
    $ jupyter notebook examples/introduction.ipynb --allow-root --ip=0.0.0.0

Finally, we would need to copy the url shown in the docker to our browser.


Contributing
############

We are open to collaboration! If you want to start contributing you only need to:

1. Check the `contributing guidelines <https://github.com/OpenMined/SyMPC/blob/main/CONTRIBUTING.md>`_.
2. Search for an issue in which you would like to work. Issues for newcomers are labeled with **good first issue**.
3. Create a PR solving the issue.


License
#######

This project is licensed under the `MIT License <https://github.com/OpenMined/SyMPC/blob/main/LICENSE.txt>`_.


Disclaimer
##########

This library should not be used in a production environment because it is still a prototype.
