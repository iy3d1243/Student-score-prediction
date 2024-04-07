import numpy as np 
import time




#the goal will be to predict the value of the score of the student 
def model(w0,w1,w2,b,x0,x1,x2):#the model function
    return w0*x0+w1*x1+w2*x2+b



def cost_function(w0,w1,w2,b,data_vect):
    n=data_vect.shape[0]
    cost = 0
    for i in range(n):
        x0, x1, x2, y = data_vect[i]
        cost += (model(w0, w1, w2, b, x0, x1, x2) - y) ** 2
    return cost/(2*n)
    


def calc_gradient(w0,w1,w2,b,data_vect):
    n=data_vect.shape[0]
    dj_dw0=0
    dj_dw1=0
    dj_dw2=0
    dj_db=0
    for i in range(n):
        x0, x1, x2, y = data_vect[i]
        prediction = model(w0, w1, w2, b, x0, x1, x2)
        dj_dw0 += (prediction - y) * x0
        dj_dw1 += (prediction - y) * x1
        dj_dw2 += (prediction - y) * x2
        dj_db += (prediction - y)
    
    return dj_dw0/n,dj_dw1/n,dj_dw2/n,dj_db/n



def gradient_descent(data_vect,q,iter):
    #q is the learning rate 
    #iter is the number of ieration for The GD
    w0=0 #initializing the model parameters on 0 
    w1=0
    w2=0
    b=0
    cost_fun_graph= {}
    for k in range(iter):
        dj_dw0,dj_dw1,dj_dw2,dj_db = calc_gradient(w0,w1,w2,b,data_vect)
        w0=w0-q*dj_dw0
        w1=w1-q*dj_dw1
        w2=w2-q*dj_dw2
        b=b-q*dj_db
        if  k%10 == 0 or k==iter-1:
            cost_fun_graph[k]=cost_function(w0,w1,w2,b,data_vect)
            print(f"gradient desent ieration num :{k} ,cost = {cost_function(w0,w1,w2,b,data_vect)}")

    return np.round(w0,2),np.round(w1,2),np.round(w2,2),np.round(b,2),cost_fun_graph


