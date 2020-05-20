RLMM
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/aclyde11/RLMM.svg?branch=master)](https://travis-ci.com/aclyde11/RLMM)
[![codecov](https://codecov.io/gh/aclyde11/RLMM/branch/master/graph/badge.svg)](https://codecov.io/gh/aclyde11/RLMM/branch/master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/aclyde11/RLMM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/aclyde11/RLMM/alerts/)

RLMM is a reinforcement learning env for molecular modeling (currently only protein-ligand docking).

## For Contributors:

### Setting up env

Make sure you are using conda, and that is your base python. You can check by
```
which python
```
which should show the base python from conda 
so for me it shows /Users/austin/anaconda3/bin/python.

Then run
```
python devtools/scripts/create_conda_env.py -n=test -p=3.6 devtools/conda-envs/test_env.yaml
conda activate test
python devtools/scripts/create_pip_installs.py -i devtools/conda-envs/test_env.yaml
```
and you should be all set. There is no need to run setup.py when working on deveolp. Not sure if that even works.


There are two ways to contribute to this project. If you are added to the project as a collaborator, please follow the steps in "Using Branch" section. Otherwise, you will have to use forks. The most important rule here is that we only use pull request to contribute and we never push directy to the master or develop branch.

### Using Branch:
1. Clone the repository: `git clone git@github.com:aclyde11/RLMM.git`.
2. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to the remote repo (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push --set-upstream origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest change from the remote develop branch to your own local develop branch: 1) `git checkout develop` 2) `git pull`.
6. Now that your local develop branch is up to date, you can update your own feature branch by: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop`.
7. Update your own feature branch on the remote repository by: `git push origin your-own-feature-branch`
8. Make a pull request with base being develop and compare being your-own-feature-branch
9. After the pull request is merged, your-own-feature-branch on the remote repository will be soon deleted, delete it on your local repository by: `git branch -d your-own-feature-branch`



#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
