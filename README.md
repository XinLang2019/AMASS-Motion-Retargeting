## Install 

step 1: create conda env
```sh
conda create -n motion_retarget python=3.8

conda activate motion_retarget
```

step 2: download this package 
```sh
git clone 
```

step3: install this package

```sh
pip install -e .

pip install -r requirement.txt
```

## AMASS Dataset Preparation
Download [AMASS Dataset](https://amass.is.tue.mpg.de/index.html) with `SMPL + H G` format and put it under `human2humanoid/data/AMASS/AMASS_Complete/`:
```
|-- human2humanoid
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD.tar.bz2
               |-- BMLhandball.tar.bz2
               |-- BMLmovi.tar.bz2
               |-- BMLrub.tar
               |-- CMU.tar.bz2
               |-- ...
               |-- Transitions.tar.bz2

```

And then `cd motion_retargeting/data/AMASS/AMASS_Complete` extract all the motion files by running:
```
for file in *.tar.bz2; do
    tar -xvjf "$file"
done
```

Then you should have:
```
|-- motion_retargeting
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD
               |-- BioMotionLab_NTroje
               |-- BMLhandball
               |-- BMLmovi
               |-- CMU
               |-- ...
               |-- Transitions

```

## SMPL Model Preparation

Download [SMPL](https://smpl.is.tue.mpg.de/download.php) with `pkl` format and put it under `motion_retargeting/data/smpl/`, and you should have:
```
|-- motion_retargeting
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0.zip
```

Then `cd motion_retargeting/data/smpl` and  `unzip SMPL_python_v.1.1.0.zip`, you should have 
```
|-- motion_retargeting
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0
            |-- models
               |-- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
            |-- smpl_webuser
            |-- ...
```
Rename these three pkl files and move it under smpl like this:
```
|-- motion_retargeting
   |-- data
      |-- smpl
         |-- SMPL_FEMALE.pkl
         |-- SMPL_MALE.pkl
         |-- SMPL_NEUTRAL.pkl
```

## Use this code

this code can convert AMASS data as `.pkl`, which can be used to train policy with [ASAP](https://github.com/LeCAR-Lab/ASAP.git).

First, you need to run:
```sh
python scripts/grad_fit_robot_shape.py --robot_type=g1
```

Second, you nedd to run:
```sh
python scripts/motion_retarget.py --robot_type=g1
```
Finally, you can run:
```sh
python scripts/motion_visually.py --robot_type=g1
```
to visual retargeting motion in Issacgym. `robot_type` can choose `g1`or `h1`

