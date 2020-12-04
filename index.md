## Welcome to Tengteng Tao (Tony)'s Personal Page

My name is Tengteng Tao (Tony). 
This is my personal page. 
I uploaded all my important things, including my program, achievenments, awards, papers, etc., to this page. 
Welcome to my pages and check all my staff!

### Awards

![Image text](https://raw.githubusercontent.com/TengtengTao/tonytao.github.io/gh-pages/image_folder/Mathematical%20Contest%20In%20Modeling.jpg)

Because of my great passion in data science and my rich experience in data analysis. I decided to participate in the 2019 Mathematical Contest in Modeling.
For the 2019 MCM, as a team leader, I chose the topic on the distribution of drug in United States, knowing that this was already an urgent issue. Due to great similarities between drug and disease, we decided to use the classic mathematical model of epidemic diseases (SEIR) as the foundation. In order to understand, explain, predict, and prevent the spreading process of drug, we read and studied more than 50 machine learning papers on drug spreading within two days. In our final paper, we used both the SEIR model (to predict the future distribution), and knowledge in linear algebra, like matrix diagonalizing and LASSO recursion（to find the most related factors）, which enabled us to run our prediction model with both high accuracy and explainability. Based on a reasonable and efficient model, our final program could both analyze current and future distribution quite correctly, which could also potentially give us innovative ways to cut off the drug spreading paths and reduce drug abuse. Our work and contribution brought us the award: Meritorious Winner(7.09%) which was a confirmation and praise on our work to our community and society.


###  Some prgrams and codes for MCM 
#### Transfer data from csv file to matrix
```markdown
import csv
import numpy as np

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData

result = []
tichu = set()
filename = r'ACS_10_5YR_DP02_with_ann.csv'


with open(filename) as f:
    reader = csv.reader(f)
    list_data = list(reader)
    for i in range(3,len(list_data[0])):
        col = []
        count = 0
        for j in range(2,len(list_data)): 
            if str.isdigit(list_data[j][i]):
                count+=1
            col.append(list_data[j][i])
        if count > len(list_data)/2:
            result.append(col)
        else:
            tichu.add(list_data[0][i])


for i in range(0,len(result)):
    total = 0 
    for j in range(0,len(result[i])):
        if not str.isdigit(result[i][j]):
            result[i][j] = total/(j+1)
        result[i][j] = float(result[i][j])
        total += result[i][j]


data = []
for i in range(0,len(result[0])):
    col = []
    for j in range(0,len(result)):
        col.append(result[j][i])
    data.append(col)
print(data)
print(len(data[0]))

percentage = 0.99
newData = zeroMean(data)
covMat = np.cov(newData,rowvar=0)
#print(covMat)
#print(len(covMat))
```

#### Codes for SIR model
```markdown
import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score


def k_fuc(p, I):
    B, r = p[:2]
    N = 15681
    k = B*I - (B*I**2)/N - r*I
    return k


k_data = [0.1282, 0.4545, -0.234, 0.6326, -0.4, 1.06]
I = [39, 44, 64, 49, 80, 48]
Para = [1, 1]


result = least_squares(lambda p, x, y: (k_fuc(p, x)-y)**2, Para, args=(np.array(I), np.array(k_data)))
Para = result.x
r2 = r2_score(np.array(k_data), k_fuc(Para, np.array(I)))
print(Para)
print(r2)
print(k_fuc(Para, 39))
```

### Some programs from former research

In my several research experiences, I created a lot of programs and models. Here I shared a few codes and programs about data fitting and data analysis I have done during my college. 

#### Codes for transfering data file (txt file) to arrays
```markdown
def o_file(path):
    x, y, z = [], [], []
    for line in open(path):
        line = line.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
        z.append(float(line[2]))
    return x, y, z
```
#### Codes for categorize data according to angels(z values)
```markdowm
def oprtor(array_x, array_y, array_z):
    spe_index = [0]
    i = 0

    while i+1 < len(array_z):
        if array_z[i] == array_z[i+1]:
            i += 1
        else:
            spe_index.append(i+1)
            i += 1
    spe_index.append(len(array_z))
    x_2dem = []
    y_2dem = []
    i_x = 0
    while i_x+1 < len(spe_index):
        x_2dem.append(array_x[spe_index[i_x]:spe_index[(i_x+1)]])
        y_2dem.append(array_y[spe_index[i_x]:spe_index[(i_x+1)]])
        i_x += 1
    return x_2dem, y_2dem
```
#### Complete program for fitting RHEED file from a certain 2D material in one specific angle. 
##### This program is composed of data classification functions, fitting models, functions for data optimization, specturm analyzing functions and plot functions
```markdown
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

def o_file(path):
    x, y, z = [], [], []
    for line in open(path):
        line = line.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
        z.append(float(line[2]))
    return x, y, z

def oprtor(array_x, array_y, array_z):
    spe_index = [0]
    i = 0

    while i+1 < len(array_z):
        if array_z[i] == array_z[i+1]:
            i += 1
        else:
            spe_index.append(i+1)
            i += 1
    spe_index.append(len(array_z))
    x_2dem = []
    y_2dem = []
    i_x = 0
    while i_x+1 < len(spe_index):
        x_2dem.append(array_x[spe_index[i_x]:spe_index[(i_x+1)]])
        y_2dem.append(array_y[spe_index[i_x]:spe_index[(i_x+1)]])
        i_x += 1
    return x_2dem, y_2dem


def G_k(K, position, height, width):
    return height/(width*(np.pi/(4*np.log(2)))**(1/2)) * np.exp(((-4*np.log(2)*(K - position)**2)/(width**2)))

def f_s(K):
    As=np.array([6.9053,1.4679,5.2034,22.2151])
    Bs=np.array([1.4379,0.2536,1.5863,56.172])
    Cs=0.8669
    summation=0
    f_s_help = (K/(4*pi))**2
    for i in range(len(As)):
        summation += As[i]*exp(-Bs[i] * f_s_help)+Cs
    return summation

def f_w(K):
    Aw=np.array([29.0818,1.72029,15.43,9.2259])
    Bw=np.array([14.4327,0.3217,5.11982,57.056])
    Cw=9.8875
    summation = 0
    f_w_help = (K/(4*pi))**2
    for i in range(len(Aw)):
        summation += Aw[i]*exp(-Bw[i] * f_w_help)+Cw
    return summation


def new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                 w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d,  x1, x2, x3, x4, x5, x6, x7, x8, x9):
    x_ls = [ x1, x2, x3, x4, x5, x6, x7, x8, x9]
    w_ls = [ w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d]
    h_ls = [ h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d]
    summation = 0
    for i in range(len(x_ls)):
        summation += h_ls[i]/(w_ls[i]*(np.pi/(4*np.log(2)))**(1/2)) * np.exp(((-4*np.log(2)*((K - x_ls[i])**2))/(w_ls[i])**2))
    return summation


def F_k(p, K):
    offset, position, height, width, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d, w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9 = p[:31]
    G_k_OP = G_k(K, position, height, width)
    f_s_OP = f_s(K)
    f_w_OP = f_w(K)
    S_d_OP = new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                     w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    correct_fc = abs(f_w(0) + f_s(0)*2) ** 2
    return S_d_OP + G_k_OP + offset

file_name = '2D_Map_MoS2_formatted_K_perp=3.997_Inverse_Angstrom.txt'  #input data from different materials here
#file_name = '2D_Map_WS2_Kperp=5p9.txt'
#file_name = '2D_Map_MoS2_formatted.txt'
#file_name = '2D Mapping.txt'
x, z, y = o_file(file_name)
list_x, list_y = oprtor(x, y, z)

bound1 = [-0.1, -5., 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          -10, -8, -5, -3, 0, 1.5, 4, 6.5, 8]
bound2 = [0.1, 5., np.inf, np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          3, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          -7, -5, -3, -1, 1, 3, 6, 8, 12]


Para = [0.06/0.866, 0.073, 4.398/0.866, 10.447,
           0.093/0.866, 0.11/0.866, 0.102/0.866, 0.134/0.866, 0.28/0.866, 0.175/0.866, 0.1/0.866, 0.072/0.866, 0.043/0.866,
           2.406, 1.561, 1.008, 0.801, 0.754, 0.787, 0.889, 1.154, 1.622,
           -9.194, -7.025, -4.624, -2.264, 0, 2.259, 4.596, 6.872, 9.235]

cost = []
r2Score = []

angle = 0      # Choose the specific angle here

RDR = int(angle/1.8)

print(np.array(list_x[RDR]))
print(list_x[RDR])

for i in range(50):
    result = least_squares(lambda p, x, y: abs(F_k(p, x)-y), Para,
                           args=(np.array(list_x[RDR]), np.array(list_y[RDR])),
                           bounds=(bound1, bound2))
    Para = result.x
    cost.append(result.cost)
    error = result.jac

    r2Score.append(r2_score(np.array(list_y[RDR]), F_k(Para,np.array(list_x[RDR]))))
    print("Cost after epoch {0:2d}:s {1}".format(i+1, result.cost))
    #print("attempts", len(error))

print("Fitted parameters are: \n", Para)
print(r2Score[0])

def single_S_d(K, h, w, x):
    return h/(w*(np.pi/(4*np.log(2)))**(1/2)) * np.exp(((-4*np.log(2)*(K - x)**2)/(w)**2))

Xc = Para[1]
B = Para[2]
Sigma = Para[3]

plt.figure(1)
plt.plot(list_x[RDR], list_y[RDR], label='Observation')


plt.plot(list_x[RDR], [G_k(i, Xc, B, Sigma) + 0.06/0.866 for i in list_x[RDR]], label='Background')
y_hat = np.array([F_k(Para, i) for i in list_x[RDR]])
plt.plot(list_x[RDR], y_hat, label='Model')

for i in range(9):
    peek = 'x'+str(i+1)
    plt.plot(list_x[RDR], [single_S_d(a, Para[i+4], Para[i+13], Para[i+22]) for a in list_x[RDR]], label=peek)


plt.plot(list_x[RDR], y_hat, label='Model')
plt.legend(loc='upper right')
plt.xlabel("k ($1/\AA$) ")
plt.ylabel("Intensity")
plt.title("RHEED profile at "+str(angle) + "degree")

y_hat = np.array([F_k(Para, i) for i in list_x[RDR]])
plt.figure(2)
plt.plot(list_x[RDR], y_hat, label='Model')
plt.plot(list_x[RDR], list_y[RDR], label='Observation')
plt.legend(loc='upper right')
plt.xlabel("k ($1/\AA$) ")
plt.ylabel("Intensity")
plt.title("RHEED profile at "+str(angle) + " degree")

plt.show()
```

#### Complete program for fitting RHEED file from a certain 2D material in all angles and plotting values VS angles in polar coordinate.
##### This program is composed of data classification functions, fitting models, functions for data optimization, specturm analyzing functions, and plotting functions. 
```markdown
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, exp, sin
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import time


time_start = time.time()

def o_file(path):
    x, y, z = [], [], []
    for line in open(path):
        line = line.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
        z.append(float(line[2]))
    return x, y, z


def oprtor(array_x, array_y, array_z):
    spe_index = [0]
    i = 0

    while i+1 < len(array_z):
        if array_z[i] == array_z[i+1]:
            i += 1
        else:
            spe_index.append(i+1)
            i += 1
    spe_index.append(len(array_z))
    x_2dem = []
    y_2dem = []
    i_x = 0
    while i_x+1 < len(spe_index):
        x_2dem.append(array_x[spe_index[i_x]:spe_index[(i_x+1)]])
        y_2dem.append(array_y[spe_index[i_x]:spe_index[(i_x+1)]])
        i_x += 1
    return x_2dem, y_2dem


def G_k(K, Xc, B, Sigma):
    dee = (K-Xc)**2
    dem = 2*(Sigma**2)
    return B*exp(-dee/dem)


def f_s(K):
    As=np.array([6.9053,1.4679,5.2034,22.2151])
    Bs=np.array([1.4379,0.2536,1.5863,56.172])
    Cs=0.8669
    summation=0
    f_s_help = (K/(4*pi))**2
    for i in range(len(As)):
        summation += As[i]*exp(-Bs[i] * f_s_help)+Cs
    return summation


def f_w(K):
    Aw=np.array([29.0818,1.72029,15.43,9.2259])
    Bw=np.array([14.4327,0.3217,5.11982,57.056])
    Cw=9.8875
    summation = 0
    f_w_help = (K/(4*pi))**2
    for i in range(len(Aw)):
        summation += Aw[i]*exp(-Bw[i] * f_w_help)+Cw
    return summation

def new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                 w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d,  x1, x2, x3, x4, x5, x6, x7, x8, x9):
    x_ls = [ x1, x2, x3, x4, x5, x6, x7, x8, x9]
    w_ls = [ w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d]
    h_ls = [ h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d]
    summation = 0
    for i in range(len(x_ls)):
        summation += h_ls[i]/(w_ls[i]*(np.pi/(4*np.log(2)))**1/2) * np.exp(((-4*np.log(2)*(K - x_ls[i])**2)/(w_ls[i])**2))
    return summation

def F_k(p, K):
    Xc, B, Sigma, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d, w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9 = p[:30]
    G_k_OP = G_k(K, Xc, B,Sigma)
    f_s_OP = f_s(K)
    f_w_OP = f_w(K)
    S_d_OP = new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                     w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    correct_fc = abs(f_w(0) + f_s(0)*2) ** 2
    return S_d_OP * ((abs(f_w_OP + f_s_OP * 2))**2)/correct_fc + G_k_OP

file_name = '2D_Map_WS2_Kperp=5p9.txt'
#file_name = '2D Mapping.txt'
#file_name = "2D_Map_MoS2_formatted.txt"
#file_name = 'MoS2.txt'
x, y, z = o_file(file_name)
list_x, list_y = oprtor(x, z, y)

bound1 = [-5., 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          -7, -4, -1, 1, 2, 5, 7, 9, 10]
bound2 = [5., np.inf, np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          -4, -1, 1, 3, 5, 7, 9, 11, 14]

Para = np.array([0, 0.5, 5,
        0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.06, 0.1, 0.05,
        1.1, 1.7, 1, 3, 1, 0.6, 1, 1, 2,
        -5, -2, 0, 2, 4, 6, 8, 10, 11])

Para1 = [0.264, 0.4398, 10.447,
           0.043, 0.072, 0.1, 0.175, 0.28, 0.134, 0.102, 0.11, 0.093,
           1.622, 1.154, 0.889, 0.787, 0.754, 0.801, 1.008, 1.561, 2.406,
           -8.807, -6.638, -4.237, -1.877, 0.387, 2.646, 4.983, 7.259, 9.622]

hwhm = []
hwhm_360 = []


for RDR in range(0, 101):

    for i in range(25):
        result = least_squares(lambda p, x, y: abs(F_k(p, x)-y), Para,
                               args=(np.array(list_x[RDR]), np.array(list_y[RDR])),
                               bounds=(bound1, bound2))
        Para = result.x
       
    for i in range(len(Para)):
        if Para[i] >= 0:
            bound2[i] = Para[i]*1.3
            bound1[i] = Para[i]*0.7
        else:
            bound2[i] = (Para[i]*(0.7))
            bound1[i] = (Para[i]*(1.3))

    print(RDR)
    hwhm.append(Para[14]/2)
    handle = open('handle.txt', 'w')
    handle.write(str(hwhm))
    handle.flush()
    handle.close()

print('DOWN!')
time_end = time.time()
print(time_end-time_start)


plt.figure(1)
phi = np.arange(0, 361.8, 1.8)
plt.plot(phi, hwhm)
plt.legend(loc='upper right')
plt.xlabel("Azimuthal angle (Degree) ")
plt.ylabel("HWHM value ($\AA$)")
plt.title("HWHM values vs azimuthal angle at Kperp = 5.9 $\AA$")



plt.figure(2)
phi_angle = np.arange(0, 2*np.pi+np.pi/100, np.pi/100)
ax = plt.subplot(111, projection='polar')
ax.set_rmax(2)
ax.set_title("HWHM values vs azimuthal angle at Kperp = 5.9 $\AA$")
ax.plot(phi_angle, hwhm)
#ax.set_rlabel_position(-22.5)
#ax.set_ylabel('HWHM ($\AA$)')
ax.text(0.52, 0.25, "$\AA$")
ax.grid(True)
plt.show()
```
#### Codes for data analysis and peaks selection based on linear algebra and advanced calculus.
```markdown
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

def o_file(path):
    x, y, z = [], [], []
    for line in open(path):
        line = line.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
        z.append(float(line[2]))
    return x, y, z


def oprtor(array_x, array_y, array_z):
    spe_index = [0]
    i = 0

    while i+1 < len(array_z):
        if array_z[i] == array_z[i+1]:
            i += 1
        else:
            spe_index.append(i+1)
            i += 1
    spe_index.append(len(array_z))
    x_2dem = []
    y_2dem = []
    i_x = 0
    while i_x+1 < len(spe_index):
        x_2dem.append(array_x[spe_index[i_x]:spe_index[(i_x+1)]])
        y_2dem.append(array_y[spe_index[i_x]:spe_index[(i_x+1)]])
        i_x += 1
    return x_2dem, y_2dem


def G_k(K, position, height, width):
    return height/(width*(np.pi/(4*np.log(2)))**(1/2)) * np.exp(((-4*np.log(2)*(K - position)**2)/(width**2)))


def f_s(K):
    As=np.array([6.9053,1.4679,5.2034,22.2151])
    Bs=np.array([1.4379,0.2536,1.5863,56.172])
    Cs=0.8669
    summation=0
    f_s_help = (K/(4*pi))**2
    for i in range(len(As)):
        summation += As[i]*exp(-Bs[i] * f_s_help)+Cs
    return summation


def f_w(K):
    Aw=np.array([29.0818,1.72029,15.43,9.2259])
    Bw=np.array([14.4327,0.3217,5.11982,57.056])
    Cw=9.8875
    summation = 0
    f_w_help = (K/(4*pi))**2
    for i in range(len(Aw)):
        summation += Aw[i]*exp(-Bw[i] * f_w_help)+Cw
    return summation


def new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                 w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d,  x1, x2, x3, x4, x5, x6, x7, x8, x9):
    x_ls = [ x1, x2, x3, x4, x5, x6, x7, x8, x9]
    w_ls = [ w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d]
    h_ls = [ h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d]
    summation = 0
    for i in range(len(x_ls)):
        summation += h_ls[i]/(w_ls[i]*(np.pi/(4*np.log(2)))**(1/2)) * np.exp(((-4*np.log(2)*(K - x_ls[i])**2)/(w_ls[i])**2))
    return summation


def F_k(p, K):
    offset, position, height, width, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d, w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9 = p[:31]
    G_k_OP = G_k(K, position, height, width)
    f_s_OP = f_s(K)
    f_w_OP = f_w(K)
    S_d_OP = new_S_d(K, h1d, h2d, h3d, h4d, h5d, h6d, h7d, h8d, h9d,
                     w1d, w2d, w3d, w4d, w5d, w6d, w7d, w8d, w9d, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    correct_fc = abs(f_w(0) + f_s(0)*2) ** 2
    return S_d_OP + G_k_OP + offset


file_name = '2D_Map_WS2_Kperp=5p9.txt'
#file_name = '2D_Map_MoS2_formatted.txt'
#file_name = '2D_Map_MoS2.txt'
#file_name = '2D_Map_MoS2_formatted_K_perp=3.997_Inverse_Angstrom.txt'
#file_name = '2D Mapping.txt'
x, z, y = o_file(file_name)
list_x, list_y = oprtor(x, y, z)



bound1 = [-0.1, -5., 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0,
          #-10, -8, -5, -3, -1, 1.5, 3, 5.5, 8]
          -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
bound2 = [np.inf, 5., np.inf, np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
          #-7, -5, -3, -1, 1, 3, 6, 8, 12]
          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
Para = [0, 0, 0.2, 5,
        0.05, 0.05, 0.41, 0.05, 0.05, 0.05, 0.06, 0.05, 0.05,        # Height
        1, 1, 1, 1, 1, 1, 1, 1, 1,                                # Width
        -9, -7, -4, -2, 0, 2, 4, 6, 9]                              # Position

cost = []
r2Score = []

angle = 1.8*90
RDR = int(angle/1.8)
angle = RDR * 1.8



for i in range(40):
    result = least_squares(lambda p, x, y: abs(F_k(p, x)-y), Para,
                           args=(np.array(list_x[RDR]), np.array(list_y[RDR])),
                           bounds=(bound1, bound2))
    Para = result.x
    cost.append(result.cost)
    error = result.jac

    r2Score.append(r2_score(np.array(list_y[RDR]), F_k(Para,np.array(list_x[RDR]))))
    #print("Cost after epoch {0:2d}:s {1}".format(i+1, result.cost))
    #print("attempts", len(error))

#print("Fitted parameters are: \n", Para)
print(r2Score[0])


#y_hat = list_y[RDR]
y_hat = np.array([F_k(Para, i) for i in list_x[RDR]])


plt.figure(0)
plt.plot(list_x[RDR], y_hat, label='Model')
plt.plot(list_x[RDR], list_y[RDR], label='Observation')
plt.legend(loc='upper right')
plt.xlabel("k ($1/\AA$) ")
plt.ylabel("Intensity")
plt.title("RHEED profile at "+str(angle) + " degrees")


x = list_x[RDR]
firstder = []
fdx=[]


for i in range(len(x)-1):
    dy = (y_hat[i]-y_hat[i+1]) * -1
    dx = (x[i]-x[i+1]) * -1
    firstder.append(dy/dx)

del x[-1]
secder = []

for i in range(len(firstder)-1):
    d2y = firstder[i+1]-firstder[i]
    d2x = (x[i]-x[i+1]) * -1
    secder.append(d2y/d2x)


secx = x[0:-1]

plt.figure(2)
plt.plot(secx, secder)

#plt.figure(3)
#plt.plot(x, firstder)
#plt.show()

xs_0p=[]
xd_0p = []
deinx=[]
secinx=[]

for i in range(len(secder)-1):
    if secx[i]<=0:
        if secder[i] <= 0 and secder[i+1] >= 0:
            xs_0p.append(i)
    else:
        if secder[i] >= 0 and secder[i+1] <= 0:
            xs_0p.append(i)


for i in range(len(xs_0p)-1):
    if secx[xs_0p[i]] <= 0:
        if firstder[xs_0p[i]-1] >=0 and firstder[xs_0p[i]+1] >=0:
            secinx.append(xs_0p[i])
    else:
        if firstder[xs_0p[i]-1] <=0 and firstder[xs_0p[i]+1] <=0:
            secinx.append(xs_0p[i])

print(secinx)
'''
print(secx[468])
print(secx[614])

print(secx[723])
print(secx[893])
'''

for i in range(len(secinx)-1):
    if secx[secinx[i]]<= 0 and secx[secinx[i+1]] >=0:
        #print(secinx[i])
        deinx.append(secinx[i])
        deinx.append(secinx[i+1])

print(deinx)
#deinx[1] = 817

ma_x = x[deinx[0]:deinx[1]]
#print(ma_x)
temp_l = y_hat[deinx[0]:deinx[1]]


plt.figure(1)
plt.plot(ma_x, temp_l)
plt.xlabel("$k_{||}$ ($1/\AA$) ")
plt.ylabel("Intensity without background")
plt.title("RHEED profile at "+str(angle) + " degrees")

dy = temp_l[-1]-temp_l[0]
dx = ma_x[-1]-ma_x[0]
slope = dy/dx
b = temp_l[0] - slope * x[deinx[0]]
y_bkg = np.array([ma_x])*slope + b


ma_me_da=[]

#for i in range(len(temp_l)):
    #ma_me_da.append(temp_l[i]-y_hat[deinx[0]])

for i in range(len(temp_l)):
    ma_me_da.append(temp_l[i]-y_bkg[0][i])


hwhm_p=[]
for i in range(len(ma_me_da)-1):
    if (ma_me_da[i] - ((max(ma_me_da)-min(ma_me_da))/2)) <= 0 and  (ma_me_da[i+1] - ((max(ma_me_da)-min(ma_me_da))/2))>= 0 :
        hwhm_p.append(ma_x[i])
    elif (ma_me_da[i] - ((max(ma_me_da)-min(ma_me_da))/2)) >= 0 and  (ma_me_da[i+1] - ((max(ma_me_da)-min(ma_me_da))/2))<= 0 :
        hwhm_p.append(ma_x[i])


print(hwhm_p)
print((max(hwhm_p)-min(hwhm_p))/2)

plt.figure(5)
plt.plot(ma_x, ma_me_da)
plt.xlabel("$k_{||}$ ($1/\AA$) ")
plt.ylabel("Intensity without background")
plt.title("RHEED profile at "+str(angle) + " degrees")
plt.show()

```







