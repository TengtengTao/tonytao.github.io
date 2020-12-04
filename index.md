## Welcome to Tengteng Tao (Tony)'s Personal Page

My name is Tengteng Tao (Tony). 
This is my personal page. 
I uploaded all my important things, including my program, achievenments, awards, papers, etc., to this page. 
Welcome to my pages and check all my staff!

### Awards

![Image text](https://raw.githubusercontent.com/TengtengTao/tonytao.github.io/gh-pages/image_folder/Mathematical%20Contest%20In%20Modeling.jpg)

Because of my great passion in data science and my rich experience in data analysis. I decided to participate in the 2019 Mathematical Contest in Modeling.
For the 2019 MCM, as a team leader, I chose the topic on the distribution of drug in United States, knowing that this was already an urgent issue. Due to great similarities between drug and disease, we decided to use the classic mathematical model of epidemic diseases (SEIR) as the foundation. In order to understand, explain, predict, and prevent the spreading process of drug, we read and studied more than 50 machine learning papers on drug spreading within two days. In our final paper, we used both the SEIR model (to predict the future distribution), and knowledge in linear algebra, like matrix diagonalizing and LASSO recursion（to find the most related factors）, which enabled us to run our prediction model with both high accuracy and explainability. Based on a reasonable and efficient model, our final program could both analyze current and future distribution quite correctly, which could also potentially give us innovative ways to cut off the drug spreading paths and reduce drug abuse. Our work and contribution brought us the award: Meritorious Winner(7.09%) which was a confirmation and praise on our work to our community and society.



### Programs

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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).


### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
