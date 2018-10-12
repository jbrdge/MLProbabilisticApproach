import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from random import randint

np.random.seed(1080)

#function to select k nearest
def klist(x1,y1,data):
    sort_list = [[0]*100 for _ in range(4)]
    #sort the array based on distance from x1,y1
    #sorted using quicksort algorithm
    for i in range(100):
        sort_list[0][i] = data[0][i]
        sort_list[1][i] = data[1][i]
        sort_list[2][i] = distance(x1,y1,data[0][i],data[1][i])
        sort_list[3][i] = i
    quicksort(sort_list[3],sort_list[0],sort_list[1],sort_list[2],0,99)
    KNN = [[0]*10 for _ in range(3)]
    for i in range(10):
        KNN[0][i] = sort_list[0][i]
        KNN[1][i] = sort_list[1][i]
        KNN[2][i] = sort_list[3][i]
    return KNN
    
def quicksort(Ac,A0,A1,A2,p,r):
    if p<r:
        q = partition(Ac,A0,A1,A2,p,r)
        quicksort(Ac,A0,A1,A2,p,q-1)
        quicksort(Ac,A0,A1,A2,q+1,r)
        
def partition(Ac,A0,A1,A2,p,r):
    x = A2[r]
    i = p-1
    for j in range(p,r):
        if A2[j]<=x:
            i = i+1
            tempA2 = A2[j]
            A2[j] = A2[i]
            A2[i] = tempA2
            tempA0 = A0[j]
            A0[j] = A0[i]
            A0[i] = tempA0
            tempA1 = A1[j]
            A1[j] = A1[i]
            A1[i] = tempA1
            tempAc = Ac[j]
            Ac[j] = Ac[i]
            Ac[i] = tempAc
    temp2 = A2[r]
    A2[r] = A2[i+1]
    A2[i+1]= temp2
    temp0 = A0[r]
    A0[r] = A0[i+1]
    A0[i+1]= temp0
    temp1 = A1[r]
    A1[r] = A1[i+1]
    A1[i+1]= temp1
    tempc = Ac[r]
    Ac[r] = Ac[i+1]
    Ac[i+1]= tempc
    return i+1
    
def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

#generate a list of data points, this can be based on a function as well
mu, sigma = 0.4, 0.1 # mean and standard deviation
a1 = list(np.random.normal(mu, sigma, 50))
a2 = list(np.random.normal(mu, sigma, 50))
a3 = [0.0]*50
mu, sigma = 0.6, 0.1
b1= list(np.random.normal(mu, sigma, 50))
b2 = list(np.random.normal(mu, sigma, 50))
b3 = [1.0]*50
data= a1+b1,a2+b2,a3+b3

#exhautively determines K-Nearest Neighbor Approximation for each point on the graph
for i in list(np.linspace(-4.5,4.5,400)):
    for j in list(np.linspace(-4.5,4.5,400)):
        X.append(i)
        Y.append(j)
        KNN = klist(i,j,data)
        for k in KNN[2]:
            p+= k/30.
        C.append(p)
        p=0
    
fig = plt.figure(figsize=(3, 3), dpi=300)
ax = plt.axes()

fig = plt.figure()
ax = plt.axes()

ax.scatter(X, Y, c=C, cmap='PRGn',s=1);
fig.savefig('foo.png', bbox_inches='tight') #saves image
#ax.scatter(data[0], data[1], c=data[2], cmap='bwr',s=20); #Plots the original dataset
#ax.scatter(0.4, 0.2, c='green',s=40); #tests the KNN for (0.4,0.2)
#KNN = klist(0.4,0.2,data) #KNN list for (0.4,0.2)
#ax.scatter(KNN[0], KNN[1], c='black',s=20); #Plot KNN for (0.4,0.2)
