https://github.com/MattYu/n-gram.git

# How to Run the Program

1- Extract the files into a directory

2- Open a terminal and navigate to the directory of step 1

3- Run the following command:

  > python nb_classifier.py V=2 N=2 D=0.01 TEST=test.txt TRAIN=train.txt
    
The values of V (0, 1, 2), N (1, 2, 3) are integers, D is a number between 0 and 1, TEST and TRAIN are the files for testing and training

3.1- Use V=3 to Run the BYOM :

  > python nb_classifier.py V=3 TEST=test.txt TRAIN=train.txt

3.2- To find the optimized hyperparameters for the BYOM, run the following command:

  > python hparam_optimizer.py 
