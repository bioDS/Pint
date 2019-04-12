# Coordinate Descent Algorithm Solving LASSO problem

## Sha Li

### Overview:

  This is a demo of using coordinate descent algorithm (including both cyclic coordinate descent 
  and randomized coordinate descent) to solve the LASSO problem, that is the `l1-regularized 
  least-squares regression problem.

  Both simulated and real world data will be used for demo training process and performances.

  The coordinate descent algorithm results will be compared with scikit-learn results on the user's
  choice of simulate data or realworld data. 

  LASSO loss function: 

  ![alt text](https://github.com/sliwhu/Coordinate-descent-algorithm-LASSO/blob/master/img/LASSO.jpg)

  Coordinate descent algorithm:
  
  ![alt text](https://github.com/sliwhu/Coordinate-descent-algorithm-LASSO/blob/master/img/algorithm.jpg)

### Organization of the project
```
Coordinate-descent-algorithm-LASSO/
  |- README.md
  |- src/
     |- demo_simulated_data.py
     |- demo_real_world_data.py
     |- demo_real_world_data.py
  |- img/
     |- LASSO.jpg
     |- algorithm.jpg
```

### User guide
Users may download and directly call the 3 demo.py files in the src folder to view training processes and performances:

* python demo_simulated_data.py
* python demo_real_world_data.py
* python demo_compare_w_sklearn.py

#### A default simulated data sets is given and a default real world data set (Hitters: 
https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv) is given.

#### Users may use default parameters and view the process directly or may self-tune parameters such as lambda, max iteration numbers and the coefficients used for simulating data sets. Please follow the instructions in the demo files and change the '__main__' function code accordinly. 

* Example 1 (demo_real_world_data.py):
if __name__=='__main__':
  algorithm = Algorithm(lambduh=2, max_iter=500)

* Example 2 (demo_simulated_data.py):
if __name__=='__main__':
  algorithm = Algorithm(lambduh=2, max_iter=500, beta0=6, beta1=7, beta2=8, beta3=9)

#### The process is very straightforward. 


### Required packages
* numpy
* pandas
* sklearn
* matplotlib
