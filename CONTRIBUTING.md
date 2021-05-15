# Contribution Guidelines

We follow the [Contribution Guidelines of PySyft](https://github.com/OpenMined/PySyft/blob/dev/CONTRIBUTING.md), please refer to them to start developing!


- [Contribution Guidelines](#contribution-guidelines)
  - [Issues / PRs](#issues--prs)
    - [Beginner Issues](#beginner-issues)
  - [Requirements](#requirements)
    - [Linux](#linux)
    - [MacOS](#macos)
    - [Windows](#windows)
  - [Git](#git)
    - [MacOS](#macos-1)
  - [Python Versions](#python-versions)
    - [MacOS](#macos-2)
    - [For Linux/WSL based Systems](#for-linuxwsl-based-systems)
    - [Using pyenv](#using-pyenv)
  - [Virtual Environments](#virtual-environments)
    - [What is a Virtual Environment](#what-is-a-virtual-environment)
      - [What about Python Package Management](#what-about-python-package-management)
  - [Pipenv](#pipenv)
    - [Common Issues](#common-issues)
  - [Git Repo](#git-repo)
    - [Forking SyMPC](#forking-sympc)
    - [Clone GitHub Repo](#clone-github-repo)
    - [Branching](#branching)
    - [Syncing your Fork](#syncing-your-fork)
    - [Learn More Git](#learn-more-git)
  - [Setting up the VirtualEnv](#setting-up-the-virtualenv)
    - [Pipenv](#pipenv-1)
    - [Install Python Dependencies](#install-python-dependencies)
    - [Linking the SyMPC src](#linking-the-sympc-src)
  - [Code Quality](#code-quality)
    - [Formatting and Linting](#formatting-and-linting)
    - [Tests and CI](#tests-and-ci)
    - [Writing Test Cases](#writing-test-cases)
    - [Documentation and Code Style Guide](#documentation-and-code-style-guide)
    - [Imports Formatting](#imports-formatting)
    - [Generating Documentation](#generating-documentation)
    - [Common Security Issues](#common-security-issues)
  - [Pre-Commit](#pre-commit)
    - [MacOS](#macos-3)
    - [Creating a Pull Request](#creating-a-pull-request)
    - [Check CI and Wait for Reviews](#check-ci-and-wait-for-reviews)
  - [Support](#support)

## Issues / PRs

- Development is done on the `main` branch.
- PR's may be closed without warning for any of the following reasons: lack of an associated Github issue, lack of tests, lack of proper documentation, or anything that isn't within the intended development roadmap of the Syft core team.
- If you are working on an existing issue posted by someone else, please ask to be added as Assignee so that effort is not duplicated.
- If you want to contribute to an issue someone else is already working on please get in contact with that person via slack or GitHub and discuss your collaboration.
- If you wish to create your own issue or PR please explain your reasoning within the Issue template and make sure your code passes all the CI checks.

**Caution**: We try our best to keep the assignee up-to-date, but as we are all humans with our own schedules mistakes happen. If you are unsure, please check the comments of the issue to see if someone else has already started work before you begin.

### Beginner Issues

If you are new to the project and want to get into the code, we recommend picking an issue with the label "good first issue". These issues should only require general programming knowledge and little to none insights into the project.

## Requirements

Before you get started you will need a few things installed depending on your operating system.

- OS Package Manager
- Python 3.6+
- git

### OSes

We intend to provide first class support for dev setup in the current versions of:

- 🐧 Ubuntu
- 🍎 MacOS
- 💠 Windows

If there are missing instructions on setup for a specific operating system or tool please open a PR.

### Linux

If you are using Ubuntu this is `apt-get` and should already be available on your machine.

### MacOS

On macOS, the main package manager is called [Brew](https://brew.sh/).

Install Brew with:

```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

Afterwards, you can now use the `brew` package manager for installing additionally required packages below.

### Windows

For Windows users, the recommended package manager is [chocolatey](https://chocolatey.org/).
Alternatively, you can use [Windows Subsystem Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and follow the instructions under Linux section.

## Git

You will need git to clone, commit and push code to GitHub.

### Linux

```
sudo apt install git
```

### MacOS

```
$ brew install git
```

## Python Versions

This project supports Python 3.6+, however, if you are contributing it can help to be able to switch between python versions to fix issues or bugs that relate to a specific python version. Depending on your operating system there are a number of ways to install different versions of python however one of the easiest is with the `pyenv` tool. Additionally, as we will be frequently be installing and changing python packages for this project we should isolate it from your system python and other projects you have using a virtualenv.

### MacOS

Install the `pyenv` tool with `brew`:

```
$ brew install pyenv
```

### For Linux/WSL based Systems

Follow the instructions provided in the repo : https://github.com/pyenv/pyenv-installer

### Using pyenv

Running the command will give you help:

```
$ pyenv
```

Lets say you wanted to install python 3.6.9 because its the version that Google Colab uses and you want to debug a Colab issue.

First, search for available python versions:

```
$  pyenv install --list | grep 3.6
...
3.6.7
3.6.8
3.6.9
3.6.10
3.6.11
3.6.12
```

Wow, there are lots of options, lets install 3.6.

```
$ pyenv install 3.6.9
```

Now, lets see what versions are installed:

```
$ pyenv versions
3.5.9
3.6.9
3.7.8
3.9.0
```

That’s all we need for now. You generally should not change which python version your system is using by default and instead we will use virtualenv manager to pick from these compiled and installed Python versions later.

## Virtual Environments

If you do not fully understand what a Virtual Environment is and why you need it, I would urge you to read this section because its actually a very simple concept but misunderstanding Python, site-packages and virtual environments lead to many common problems when working with projects and packages.

### What is a Virtual Environment

Ever wonder how python finds your packages that you have installed? The simple answer is, it recursively searches up a few folders from where ever the binary `python` or `python.exe` looking for a folder called site-packages.

When you open a shell try typing:

```
$ which python
/usr/local/bin/python3
```

Lets take a closer look at that symlink:

```
$ ls -l /usr/local/bin/python3
/usr/local/bin/python3 -> ../Cellar/python@3.9/3.9.0_1/bin/python3
```

Okay, so that means if I run this python3 interpreter I’m going to get python 3.9.0 and it will look for packages where ever that folder is in my Brew Cellar.

So what if I wanted to isolate a project from that and even use a different version of python you ask?
Quite simply a virtual environment is a folder where you store a copy of the python binary you want to use, and then you change the PATH of your shell to use that binary first so all future package resolution commands including installing packages with `pip` etc will go in that subfolder. This explains why with most virtualenv tools you have to activate them often by running `source` on a shell file to change your shells PATH.

This is so common there is a multitude of tools to help with this, and the process is now officially supported within python3 itself.

**Bonus Points**
Watch: Reverse-engineering Ian Bicking's brain: inside pip and virtualenv
https://www.youtube.com/watch?v=DlTasnqVldc

##### What about Python Package Management

Okay so virtualenvs are only part of the process, they give you isolated folder structures in which you can install, update and delete packages without worrying about messing up other projects. But how do I install a package? Is that only pip, what about conda or pipenv or poetry?

Most of these tools aim to provide the same functionality which is to create virtualenvs, and handle the installation of packages as well as making the experience of activating and managing virtualenvs as seamless as possible. Some, as in the case of conda even provide their own package repositories and additional non-python package support.

For the example below I will be using `pipenv` purely because it is extremely simple to use, and is itself simply a pip package which means as long as you have any version of python3 on your system you can use this to bootstrap everything else.

| name       | packages | virtualenvs |
| ---------- | -------- | ----------- |
| pip + venv | ✅       | ✅          |
| pipenv     | ✅       | ✅          |
| conda      | ✅       | ✅          |
| poetry     | ✅       | ✅          |

## Pipenv

As you will be running pipenv to create virtualenvs you will want to install pipenv into your normal system python site-packages.
This can be achieved by simply `pip` installing it from your shell.

```
$ pip install pipenv
```

### Common Issues

- what is the difference between pip and pip3?
  pip3 was introduced as an alias to use the pip package manager from python3 on systems where python 2.x is still used by the operating system.
  When in doubt use pip3 or check the path and version that your python or pip binary is using.
- I don't have pip?
  On some systems like Ubuntu, you need to install pip first with `apt-get install python3-pip` or you can use the new official way to install pip from python:

```
$ python3 -m ensurepip
```

## Git Repo

### Forking SyMPC

As you will be making contributions you will need somewhere to push your code. The way you do this is by forking the repository so that your own GitHub user profile has a copy of the source code.

Navigate to the page and click the fork button:
https://github.com/OpenMined/SyMPC

You will now have a URL like this with your copy:
https://github.com/\<your-username>/SyMPC

### Clone GitHub Repo

```
$ git clone https://github.com/<your-username>/SyMPC
$ cd SyMPC
```

### Branching

Do not forget to create a branch from `main` that describes the issue or feature you are working on.

```
$ git checkout -b "feature_1234"
```

### Syncing your Fork

To sync your fork (remote) with the OpenMined/SyMPC (upstream) repository please see this [Guide](https://help.github.com/articles/syncing-a-fork/) on how to sync your fork or follow the given commands.

```
$ git remote update
$ git checkout <branch-name>
$ git rebase upstream/<branch-name>
```

### Learn More Git

If you want to learn more about git or Github then check out [this guide](https://guides.github.com/activities/hello-world).

## Setting up the VirtualEnv

Lets create a virtualenv and install the required packages so that we can start developing on Syft.

### Pipenv

Using pipenv you would do the following:

```
$ pipenv --python=3.6
```

We installed python 3.6 earlier so here we can just specify the version and we will get a virtualenv with that version. If you want to use a different version make sure to install it to your system with your system package manager or `pyenv` first.

We have created the virtualenv but it is not active yet.
If you type the following:

```
$ which python
/usr/bin/python
```

You can see that we still have a python path that is in our system binary folder.

Lets activate the virtualenv with:

```
$ pipenv shell
```

You should now see that the prompt has changed and if you run the following:

```
$ which python
/Users/madhavajay/.local/share/virtualenvs/PySyft-lHlz_cKe/bin/python
```

Okay, any time we are inside the virtualenv every python and pip command we run will use this isolated version that we defined and will not affect the rest of the system or other projects.

### Install Python Dependencies

Once you are inside the virtualenv you can do this with pip or pipenv.

#### Windows

To install the dependencies properly on Windows, you must first install PyTorch because most Windows binary wheels are not available on PyPI.

You can do this by telling `pip` to use the official PyTorch Wheel Repository instead of the default PyPI repository.

```
$ pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

_Note_ If you need a specific version you can supply it with the `==` syntax. Also be aware there are different versions depending on if you require CUDA for GPU usage or CPU only such as what we use in GitHub CI.

```
$ pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Then continue below with requirements.dev.txt like normal.

**NOTE** this is required for several `dev` packages like pytest-xdist etc.

```
$ pip install -r requirements.dev.txt
```

or

```
$ pipenv install --dev --skip-lock
```

Now you can verify we have installed a lot of stuff by running:

```
$ pip freeze
```

### Linking the SyMPC src

Now we need to link the src directory of the SyMPC code base into our site-packages
so that it acts like it’s installed but we can change any file we like and `import` again
to see the changes.

```
$ pip install -e .
```

The best way to know everything is working is to run the tests.

Run the quick tests with all your CPU cores by running:

```
$ pytest -n auto
```

If they pass then you know everything is set up correctly.


### Linking the PySyft src

SyMPC is a companion library for PySyft. For the moment they are highly coupled.
One of the SyMPC goals is to convert it into a standalone library. Thus, PySyft
will depend on SyMPC, but not the other way around. This point is not yet reached
and this adds some complications when applying some changes that in SyMPC that requires changes in PySyft to work.

**Example**: You implement a new function on a ShareTensor in SyMPC. In your test, you call that function from a PySyft object pointer `share_pointer_in_pysyft.new_method()`.  Clearly, this test will not work until PySyft uses this new SyMPC version you are developing.

To test this, you will need to clone the PySyft library, install the dependencies and link the PySyft src as we have done with SyMPC. For more detailed information, please visit the [PySyft contributin guide](https://github.com/OpenMined/PySyft/blob/dev/packages/syft/CONTRIBUTING.md).

As a little summary, you need to:

```bash
$ git clone https://github.com/OpenMined/PySyft.git
$ cd PySyft/packages
$ pip install -r 'requirements.txt'
$ pip install -e syft
```

This steps are here as a guide. Please, check the [PySyft contributin guide](https://github.com/OpenMined/PySyft/blob/dev/packages/syft/CONTRIBUTING.md) for the detailed steps.

Now, you are prepared to face this kind of changes. A normal change in SyMPC looks like:

1. Change SyMPC code
2. Run the tests

A change that implies both libraries looks like:

1. Change SyMPC code
2. Change PySyft code
3. Install SyMPC with `pip install -e .`
4. Install PySyft with `pip install -e syft`
5. Run SyMPC tests
6. Run PySyft tests

## Code Quality

### Formatting and Linting

We use several tools to keep our codebase high quality.

- black
- flake8
- isort

### Tests and CI

When you push your code it will run through a series of GitHub Actions which will ensure that the code meets our minimum standards of code quality before a Pull Request can be reviewed and approved.

To make sure your code will pass these CI checks before you push you should use the pre-commit hooks and run tests locally.

We aim to have a 100% test coverage, and the GitHub Actions CI will fail if the coverage is below a certain value. You can evaluate your coverage using the following commands.

```
$ pytest -m fast -n auto
```

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage at 100%. You can always ask for help in slack or via GitHub if you don't feel confident about your tests.

### Documentation and Code Style Guide

To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation, we are using the Google Python Style Rules which can be found [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). A well-written example can we viewed [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Your documentation should not describe the obvious, but explain what's the intention behind the code and how you tried to realize your intention.

You should also document non-self-explanatory code fragments e.g. complicated for-loops. Again please do not just describe what each line is doing but also explain the idea behind the code fragment and why you decided to use that exact solution.

You can check documentation style errors with:

```bash
# Check if the description matches a function signature
$ darglint .
# Checks compliance with python docstrings conventions
$ pydocstyle .
```

### Imports Formatting

We use isort to automatically format the python imports. 
Run isort manually like this:

```
$ isort .
```

### Common Security Issues

Security issues are hard to avoid if you dont know them. To avoid the most common security issues we use bandit.
We can execute:

```
$ bandit .
```

### Generating Documentation

You can execute the following command from the root directory of the project. It will:

1. Navigate to the docs folder
2. Build the html files
3. Navigate to the folder with the generated html files
4. Start a local http server

```
$ cd docs && make html && cd ../build/sphinx/html && python3 -m http.server
```

You can now navigate to http://0.0.0.0:8000/.

## Pre-Commit

We are using a tool called [pre-commit](https://pre-commit.com/) which is a plugin system that allows easy configuration of popular code quality tools such as linting, formatting, testing and security checks.

### MacOS

First, install the pre-commit tool:

```
$ brew install pre-commit
```

Now make sure to install the pre-commit hooks for this repo:

```
$ cd SyMPC
$ pre-commit install
```

To make sure its working run the pre-commit checks with:

```
$ pre-commit run --all-files
```

Now every time you try to commit code these checks will run and warn you if there was an issue.
These same checks run on CI, so if it fails on your machine, it will probably fail on GitHub.

### Creating a Pull Request

At any point in time, you can create a pull request, so others can see your changes and give you feedback. Please create all pull requests to the `main` branch.

If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title and choose the Draft option on GitHub.

Example:`[WIP] Serialization of PointerTensor`

### Check CI and Wait for Reviews

After each commit, GitHub Actions will check your new code against the formatting guidelines (should not cause any problems when you set up your pre-commit hook) and execute the tests to check if the test coverage is high enough.

We will only merge PRs that pass the GitHub Actions checks.

If your check fails, don't worry, you will still be able to make changes and make your code pass the checks. Try to replicate the issue on your local machine by running the same check or test which failed on the same version of Python if possible. Once the issue is fixed, simply push your code again to the same branch and the PR will automatically update and rerun CI.

## Support

For support in contributing to this project and like to follow along with any code changes to the library, please join the #code_sympc and #cryptography Slack channel. [Click here to join our Slack community!](https://slack.openmined.org/)
