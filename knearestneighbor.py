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
mu, sigma = 1.5, .085
e1= list(np.random.normal(mu, sigma, 50))
e2 = list(np.random.normal(mu, sigma, 50))
e3 = [2.0]*50

f1=[]
f2=[]
for x in range(0,50):
    f1.append([-1.5+2*np.cos(x*np.pi/25)])
    f2.append([1.5+2*np.sin(x*np.pi/25)])
f3 = [-2.0]*50

g1=[]
g2=[]
for x in range(0,50):
    g1.append([1.5+2*np.cos(x*np.pi/25)+4*np.sin(x*np.pi/250)])
    g2.append([-1.5+2*np.sin(x*np.pi/25)])
g3 = [-1.0]*50

a1=[]
a2=[]
for x in range(0,50):
    a1.append([1.5+2*np.cos(x*np.pi/25)])
    a2.append([-1.5+2*np.sin(x*np.pi/25)+4*np.sin(x*np.pi/250)])
a3 = [1.0]*50

data=a1+e1+f1+g1,a2+e2+f2+g2,a3+e3+f3+g3
#data= a1+b1+c1+d1+e1+f1,a2+b2+c2+d2+e2+f2,a3+b3+c3+d3+e3+f3

X=[]
Y=[]
C=[]
p=0.

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

ax.scatter(X, Y, c=C, cmap='PRGn',s=1); #Plots KNN Map
#ax.scatter(data[0], data[1], c=data[2], cmap='bwr',s=20); #Plots the original dataset
#ax.scatter(0.4, 0.2, c='green',s=40); #tests the KNN for (0.4,0.2)
#KNN = klist(0.4,0.2,data) #KNN list for (0.4,0.2)
#ax.scatter(KNN[0], KNN[1], c='black',s=20); #Plot KNN for (0.4,0.2)
fig.savefig('foo.png', bbox_inches='tight') #saves image
