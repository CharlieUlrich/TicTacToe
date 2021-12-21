#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Importing the datasets
import numpy
numpy.random.seed(0)
fin = numpy.loadtxt('../tictac_final.txt')
finX = fin[:,:9]
finY = fin[:,9:]

sin = numpy.loadtxt('../tictac_single.txt')
sinX = sin[:,:9]
sinY = sin[:,9:]

mul = numpy.loadtxt('../tictac_multi.txt')
mulX = mul[:,:9]
mulY = mul[:,9:]


# In[2]:


###DECIDING WHAT TO DO
watToDo = input("To view regressor and classifier results enter 1, to play tictactoe enter 2, for both enter 3")
whatToDo = int(watToDo)


# In[3]:


###CLASSIFIERS
#Importing k-folds model, linear svm classification model, and metrics
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Creating KFold
kf = KFold(n_splits=10, shuffle=True)

#linear svm classifier
from sklearn.svm import LinearSVC
linearSVM = LinearSVC()

#Loop through kfolds TICTACFINAL
kcnt = 1
avgsum = 0
for train,test in kf.split(finX):
    x_train, x_test = finX[train], finX[test]
    y_train, y_test = finY[train], finY[test]
    linearSVM.fit(x_train,y_train)
    predictionFinSVM = linearSVM.predict(x_test)
    #Print accuracy and confusion matrix, format is:
    #[true negatives (TN), false positives (FP)]
    #[false negatives (FN), true positives (TP)]
    if whatToDo != 2:
        print("\n","KFOLD ", kcnt)
        print("TICTACFINAL Linear SVM Accuracy:", metrics.accuracy_score(y_test,predictionFinSVM))
    avgsum += metrics.accuracy_score(y_test,predictionFinSVM)
    #Getting then normalizing the confusion matrix
    cm = confusion_matrix(predictionFinSVM,y_test)
    newcm = numpy.zeros((2,2))
    i=0
    for row in cm:
        sumv = row[0] + row[1]
        newcm[i,0] = float(cm[i,0])/float(sumv)
        newcm[i,1] = float(cm[i,1])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix:")
        print(newcm)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')

#loop through kfolds TICTAC INTERMEDIATE SINGLE
kcnt = 1
avgsum = 0
for train,test in kf.split(sinX):
    x_train, x_test = sinX[train], sinX[test]
    y_train, y_test = sinY[train], sinY[test]
    linearSVM.fit(x_train,y_train)
    predictionSinSVM = linearSVM.predict(x_test)

    #Print accuracy and confusion matrix
    if whatToDo != 2:
        print("\n", "KFOLD ", kcnt)
        print("TICTACINTSIN Linear SVM Accuracy:", metrics.accuracy_score(y_test,predictionSinSVM))
    avgsum += metrics.accuracy_score(y_test,predictionSinSVM)
    cm = confusion_matrix(predictionSinSVM,y_test)
    newcm = numpy.zeros((9,9))
    i=0
    for row in cm:
        sumv = row[0] + row[1] + row[2] + row[3]+ row[4] + row[5]+ row[6] + row[7]+ row[8]
        if(sumv!=0):
            newcm[i,0] = float(cm[i,0])/float(sumv)
            newcm[i,1] = float(cm[i,1])/float(sumv)
            newcm[i,2] = float(cm[i,2])/float(sumv)
            newcm[i,3] = float(cm[i,3])/float(sumv)
            newcm[i,4] = float(cm[i,4])/float(sumv)
            newcm[i,5] = float(cm[i,5])/float(sumv)
            newcm[i,6] = float(cm[i,6])/float(sumv)
            newcm[i,7] = float(cm[i,7])/float(sumv)
            newcm[i,8] = float(cm[i,8])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix: ")
        print(newcm)   
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')


# In[4]:


#Creating K-nearest neighbors model
from sklearn.neighbors import KNeighborsClassifier
kneighbors = KNeighborsClassifier(n_neighbors=5)

#Loop through kfolds TICTACFINAL
kcnt = 1
avgsum = 0
for train,test in kf.split(finX):
    x_train, x_test = finX[train], finX[test]
    y_train, y_test = finY[train], finY[test]
    kneighbors.fit(x_train,y_train)
    predictionFinKN = kneighbors.predict(x_test)
    #Print accuracy and confusion matrix, format is:
    #[true negatives (TN), false positives (FP)]
    #[false negatives (FN), true positives (TP)]
    if whatToDo != 2:
        print("\n", "KFOLD ", kcnt)
        print("TICTACFINAL Kneighbors Accuracy:", metrics.accuracy_score(y_test,predictionFinKN))
    avgsum+=metrics.accuracy_score(y_test,predictionFinKN)
    #Getting then normalizing the confusion matrix
    cm = confusion_matrix(predictionFinKN,y_test)
    newcm = numpy.zeros((2,2))
    i=0
    for row in cm:
        sumv = row[0] + row[1]
        newcm[i,0] = float(cm[i,0])/float(sumv)
        newcm[i,1] = float(cm[i,1])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix:")
        print(newcm)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')
                                   
#Loop through KFolds TicTac Intermediate Single 
kcnt = 1
avgsum = 0
for train,test in kf.split(sinX):
    x_train, x_test = sinX[train], sinX[test]
    y_train, y_test = sinY[train], sinY[test]
    kneighbors.fit(x_train,y_train)
    predictionSinKN = kneighbors.predict(x_test)
    #Print accuracy and confusion matrix, format is:
    #[true negatives (TN), false positives (FP)]
    #[false negatives (FN), true positives (TP)]
    if whatToDo != 2:
        print("\n", "KFOLD ", kcnt)
        print("TICTAC INTERMEDIATE SINGLE Kneighbors Accuracy:", metrics.accuracy_score(y_test,predictionSinKN))
    avgsum+=metrics.accuracy_score(y_test,predictionSinKN)
    #Getting then normalizing the confusion matrix
    cm = confusion_matrix(predictionSinKN,y_test)
    newcm = numpy.zeros((9,9))
    i=0
    for row in cm:
        sumv = row[0] + row[1] + row[2] + row[3]+ row[4] + row[5]+ row[6] + row[7]+ row[8]
        if(sumv!=0):
            newcm[i,0] = float(cm[i,0])/float(sumv)
            newcm[i,1] = float(cm[i,1])/float(sumv)
            newcm[i,2] = float(cm[i,2])/float(sumv)
            newcm[i,3] = float(cm[i,3])/float(sumv)
            newcm[i,4] = float(cm[i,4])/float(sumv)
            newcm[i,5] = float(cm[i,5])/float(sumv)
            newcm[i,6] = float(cm[i,6])/float(sumv)
            newcm[i,7] = float(cm[i,7])/float(sumv)
            newcm[i,8] = float(cm[i,8])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix: ")
        print(newcm)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')


# In[ ]:


###MULTILAYER PERCEPTRON
#Importing the Multilayer Perceptron model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver="adam")

#Training / Testing on TICTACTOEFINAL
kcnt = 1
avgsum = 0
for train,test in kf.split(finX):
    x_train, x_test = finX[train], finX[test]
    y_train, y_test = finY[train], finY[test]
    mlp.fit(x_train,y_train)
    predictionFinMLP = mlp.predict(x_test)
    #Print accuracy and confusion matrix, format is:
    #[true negatives (TN), false positives (FP)]
    #[false negatives (FN), true positives (TP)]
    if whatToDo != 2:
        print("KFOLD ", kcnt, "\n")
        print("TICTACFINAL MLP Accuracy:", metrics.accuracy_score(y_test,predictionFinMLP))
    avgsum+=metrics.accuracy_score(y_test,predictionFinMLP)
   #Getting then normalizing the confusion matrix
    cm = confusion_matrix(predictionFinMLP,y_test)
    newcm = numpy.zeros((2,2))
    i=0
    for row in cm:
        sumv = row[0] + row[1]
        newcm[i,0] = float(cm[i,0])/float(sumv)
        newcm[i,1] = float(cm[i,1])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix:")
        print(newcm)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')
    
#TRAINING / TESTING ON TICTACTOE INTERMEDIATE SINGLE
kcnt = 1
avgsum=0
for train,test in kf.split(sinX):
    x_train, x_test = sinX[train], sinX[test]
    y_train, y_test = sinY[train], sinY[test]
    mlp.fit(x_train,y_train)
    predictionSinMLP = mlp.predict(x_test)
    #Print accuracy and confusion matrix, format is:
    #[true negatives (TN), false positives (FP)]
    #[false negatives (FN), true positives (TP)]
    if whatToDo != 2:
        print("KFOLD ", kcnt, "\n")
        print("TICTAC INTERMEDIATE SINGLE MLP Accuracy:", metrics.accuracy_score(y_test,predictionSinMLP))
    avgsum+=metrics.accuracy_score(y_test,predictionSinMLP)
    #Getting then normalizing the confusion matrix
    cm = confusion_matrix(predictionSinMLP,y_test)
    newcm = numpy.zeros((9,9))
    i=0
    for row in cm:
        sumv = row[0] + row[1] + row[2] + row[3]+ row[4] + row[5]+ row[6] + row[7]+ row[8]
        if(sumv!=0):
            newcm[i,0] = float(cm[i,0])/float(sumv)
            newcm[i,1] = float(cm[i,1])/float(sumv)
            newcm[i,2] = float(cm[i,2])/float(sumv)
            newcm[i,3] = float(cm[i,3])/float(sumv)
            newcm[i,4] = float(cm[i,4])/float(sumv)
            newcm[i,5] = float(cm[i,5])/float(sumv)
            newcm[i,6] = float(cm[i,6])/float(sumv)
            newcm[i,7] = float(cm[i,7])/float(sumv)
            newcm[i,8] = float(cm[i,8])/float(sumv)
        i+=1
    if whatToDo != 2:
        print("Confusion Matrix: ")
        print(newcm)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')


# In[ ]:


###MOVING TO REGRESSORS


# In[ ]:


#K-Neighbors regression uses same model as for classification so don't need to import new stuff
kneighbors = KNeighborsClassifier(n_neighbors=5)

#Loop through kfolds TICTACFINAL
kcnt = 1
avgsum=0

for train,test in kf.split(mulX):
    x_train, x_test = mulX[train], mulX[test]
    y_train, y_test = mulY[train], mulY[test]
    kneighbors.fit(x_train,y_train)
    predictionMulKN = kneighbors.predict(x_test)
    #Print accuracy
    if whatToDo != 2:
        print("KFOLD ", kcnt, "\n")
        print("TICTACMULTI Kneighbors Accuracy:", metrics.accuracy_score(y_test,predictionMulKN))
    avgsum+=metrics.accuracy_score(y_test,predictionMulKN)
    kcnt+=1
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')


# In[ ]:


#Multilayer Perceptron Regression
mlpr = MLPClassifier(solver="lbfgs")
#Training / Testing on TICTACTOEFINAL
kcnt = 1
avgsum=0
for train,test in kf.split(mulX):
    x_train, x_test = mulX[train], mulX[test]
    y_train, y_test = mulY[train], mulY[test]
    mlpr.fit(x_train,y_train)
    predictionMulMLP = mlpr.predict(x_test)
    #Print accuracy 
    if whatToDo != 2:
        print("KFOLD ", kcnt, "\n")
        print("TICTACMULTI MLP Accuracy:", metrics.accuracy_score(y_test,predictionMulMLP))
    avgsum+=metrics.accuracy_score(y_test,predictionMulMLP)
    kcnt+=1
mlpr.predict(mulX)
if whatToDo != 2:
    print("AVERAGE ACCURACY: ", avgsum/10 * 100, "%",sep='')


# In[ ]:


from numpy import linalg
#Linear Regression Implementation
#Use normal matrix equations
#Repeating data for ease
xVals = mulX
yVals = mulY
netacc = 0
kfCnt = 1
for train,test in kf.split(mulX):
    x_train, x_test = mulX[train], mulX[test]
    y_train, y_test = mulY[train], mulY[test]
    #Creating rows which hold weights where w0 refers to x index, and index in array refers to y index
    xT =x_train.T
    weights0 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,0])
    #print(weights0)
    weights1 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,1])
    weights2 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,2])
    weights3 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,3])
    weights4 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,4])
    weights5 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,5])
    weights6 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,6])
    weights7 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,7])
    weights8 = linalg.inv(xT.dot(x_train)).dot(xT).dot(y_train[:,8])
    weights = [weights0,weights1,weights2,weights3,weights4,weights5,weights6,weights7,weights8]
    #print(weights)

    ###Accuracy Checker
    ###Check if y val = weights times x val rounded to nearest val
    acc1Sum = 0
    tot1 = 0
    acc2Sum = 0
    tot2 = 0
    for row in x_train:
        cnt = 0
        i = 0
        while i < 9:
            sumval = row[0] * weights[i][0]
            sumval += row[1] * weights[i][1]
            sumval += row[2] * weights[i][2]
            sumval += row[3] * weights[i][3]
            sumval += row[4] * weights[i][4]
            sumval += row[5] * weights[i][5]
            sumval += row[6] * weights[i][6]
            sumval += row[7] * weights[i][7]
            sumval += row[8] * weights[i][8]
            #print(weights[i])
            #Various values were tested and .2061 was found to produce the highest accuracy
            if sumval > .2061:
               # print("SUM: ", sumval, "Prediction: ", 1, "YVAL: ", yVals[cnt][i])
                if y_test[cnt][i] == 1:
                    acc1Sum +=1
                tot1 += 1
            else:
               # print("SUM: ", sumval, "Prediction: ", 0, "YVAL: ", yVals[cnt][i])
                if y_test[cnt][i] == 0:
                    acc2Sum+=1
                tot2 += 1
            i += 1
        cnt+=1
        
    if whatToDo != 2:
        print("KFOLD", kfCnt)
        print("Accuracy 1: ", acc1Sum/tot1 * 100, "%", sep = '')
        print("Accuracy 0: ", acc2Sum/tot2 * 100, "%", sep = '')
    acc = (acc1Sum/tot1 + acc2Sum/tot2)/2
    if whatToDo != 2:
        print("Average accuracy: ", acc * 100 , "% \n", sep = '')
    netacc+=acc
    kfCnt+=1
if whatToDo != 2:
    print("OVERALL AVERAGE ACCURACY: ", netacc/10 * 100 , "%", sep = '')
    #Formula to predict y = x0w0 + x1w1 + ... + x8w8
    #Weights0 has all the weights for y0 from x0 ... x8
    #for xrow x vals
        #i=0
        #for yrow y vals
            #sumval = xrow[0]*weights[i][0]
            #...
            #sumval += row


# In[ ]:


## TICTACTOE GAME
if whatToDo != 1:
    def print_board():
            for i in boardrep:
                print(i)

    def add_move(pos, player):
            ymove = pos
            if ymove%3 == 0:
                ymove=2
            elif ymove%3 ==1:
                ymove = 0
            else:
                ymove = 1
            xmove = pos
            if xmove<4:
                xmove = 0
            elif xmove <7:
                xmove = 1
            else:
                xmove = 2
            boardrep[xmove][ymove] = player

    def check_Winner():
        if board[0][0] == board[0][4] and board[0][0] == board[0][8] and board[0][0] != 0:
            winner = boardrep[0][0]
            return True
        elif board[0][2] == board[0][4] and board[0][2] == board[0][6] and board[0][2] != 0:
            winner = boardrep[0][2]        
            return True
        i = 0
        while i <= 2:
            if boardrep[i][0] == boardrep[i][1] and boardrep[i][0] == boardrep[i][2] and boardrep[i][0] != '':
                winner = boardrep[i][0]
                return True
            elif boardrep[0][i] == boardrep[1][i] and boardrep[0][i] == boardrep[2][i] and boardrep[0][i] != '':
                winner = boardrep[0][i]
                return True
            i+=1
        return False
    #Game loop
    while True:
        board = numpy.zeros((1,9))
        boardrep = [['','',''],['','',''],['','','']]


        print("Welcome to TIC TAC TOE")
        print_board()
        print("In order to play enter where you want to play as a number from 1-9 with each representing a position on the board")
        print("The Positions represented by 1-9 are:")
        positions = [[1,2,3],[4,5,6],[7,8,9]]
        for i in positions:
            print(i)
        print("You will play as X, the computer is O, good luck!")

        noWinnerYet = True
        full = 0
        while noWinnerYet:
                while True:
                    move = int(input("\nInput your move! "))
                    if board[0][move-1] == 0:
                        add_move(move,'X')
                        full += 1
                        board[0][move-1] = 1
                        print_board()
                        break
                    else:
                        print("Spot taken enter a different one!")
                if check_Winner():
                    print('WINNER WINNER CHICKEN DINNER!')
                    break
                if full == 9:
                    print("Draw!")
                    break

                #Getting CPU Move
                predMlp = mlpr.predict(board)
                predKN = kneighbors.predict(board)
                breaker = True
                i = 0
                skip = False
                useK = False
                while breaker:
                    if predMlp[0][i] == 1 and board[0][i] == 0:
                        if predMlp[0][i] == predKN[0][i] or skip :
                            board[0][i] = -1
                            add_move(i+1,'O')
                            breaker = False
                    if useK:
                        if predKN[0][i] == 1 and board[0][i] == 0:
                            board[0][i] = -1
                            add_move(i+1,'O')
                            breaker = False
                    i += 1
                    if i == 9:
                        if skip:
                            useK = True
                        skip = True
                        i = 0
                print("\n Computers Move: ")
                print_board()
                full += 1
                if check_Winner():
                    print('Tough loss')
                    break

        quit = False
        while True:
                inp = input("Play Again? Type Yes to Continue or No to Quit")
                if inp == "Yes":
                    break
                elif inp == "No":
                    quit = True
                    print("Thanks for Playing!")
                    break
                else:
                    print("Invalid Input Try Again")
        if quit:
            break


# In[ ]:




