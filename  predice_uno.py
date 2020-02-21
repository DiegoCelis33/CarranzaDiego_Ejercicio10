import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

%matplotlib inline

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))

# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Vamos a entrenar solamente con los digitos iguales a 1
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

plt.figure(figsize=(15,5))
plt.subplot(2,3,1)
plt.title("Matriz de Covarianza")
plt.imshow(cov)

plt.subplot(2,3,2)
plt.title("Varianza explicada")
plt.plot(np.cumsum(valores)/np.sum(valores))
plt.xlabel("Componentes")
plt.ylabel("Fraccion")
max_comps = (np.count_nonzero((np.cumsum(valores)/np.sum(valores))<0.6))
print(max_comps+1) # Necesito este numero de componentes para tener al menos el 60 de la varianza.

plt.subplot(2,3,4)
plt.imshow(vectores[:,0].reshape(8,8))
plt.title('Primer Eigenvector')
plt.subplot(2,3,5)
plt.title('Segundo Eigenvector')
plt.imshow(vectores[:,1].reshape(8,8))
plt.subplot(2,3,6)
plt.title('Tercer Eigenvector')
plt.imshow(vectores[:,2].reshape(8,8))
plt.subplots_adjust(hspace=0.5)

x_train_2 = x_train[dd]

x_train_3 = x_train[dd]

x_nueva_base = np.matmul(x_train_3,vectores)

plt.scatter(x_nueva_base[:, 0], x_nueva_base[:, 1], cmap="Paired")


#numero_test = 1
#dd_test = y_test==numero_test

cov_test = np.cov(x_test.T)
valores_test, vectores_test = np.linalg.eig(cov_test)
valores_test = np.real(valores_test)
vectores_test = np.real(vectores_test)
ii = np.argsort(-valores_test)
valores_test = valores_test[ii]
vectores_test = vectores_test[:,ii]

plt.figure(figsize=(15,5))
plt.subplot(2,3,1)
plt.title("Matriz de Covarianza test")
plt.imshow(cov_test)

plt.subplot(2,3,4)
plt.imshow(vectores_test[:,0].reshape(8,8))
plt.title('Primer Eigenvector')
plt.subplot(2,3,5)
plt.title('Segundo Eigenvector')
plt.imshow(vectores_test[:,1].reshape(8,8))
plt.subplot(2,3,6)
plt.title('Tercer Eigenvector')
plt.imshow(vectores_test[:,2].reshape(8,8))
plt.subplots_adjust(hspace=0.5)

print(vectores.shape)


x_test_2 = x_test#[dd_test]

x_nueva_base_test = np.matmul(x_test_2,vectores_test)
plt.figure()
plt.scatter(x_nueva_base_test[:, 0], x_nueva_base_test[:, 1], cmap="Paired")

plt.scatter(x_nueva_base_test[:, 0], x_nueva_base_test[:, 1], label='test')
plt.scatter(x_nueva_base[:, 0], x_nueva_base[:, 1], label='train 1')
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.legend()

beta_train_uno_x = np.mean(x_nueva_base[:,0])
SD_beta_train_uno_x = np.std(x_nueva_base[:,0])

beta_train_uno_y = np.mean(x_nueva_base[:,1])
SD_beta_train_uno_y = np.std(x_nueva_base[:,1])


matriz_verdad_test = np.zeros(len(y_test))


for i in range(len(y_test)):
    matriz_verdad_test_p = 0
    if((beta_train_uno_x-SD_beta_train_uno_x)< x_nueva_base_test[i,0] < (beta_train_uno_x+SD_beta_train_uno_x)):
        if((beta_train_uno_y-SD_beta_train_uno_y)< x_nueva_base_test[i,1] < (beta_train_uno_y+SD_beta_train_uno_y)):
            matriz_verdad_test[i] = 1
            
        
        else:
            matriz_verdad_test[i] = 0

from sklearn.metrics import confusion_matrix

numero = 1

y_test == numero

y_test_uno = y_test


y_true = y_test_uno
y_predict = matriz_verdad_test

confusion = confusion_matrix(y_true, y_predict)

TP_list = np.zeros(len(y_test))
FN_list = np.zeros(len(y_test))
FP_list = np.zeros(len(y_test))
TN_list = np.zeros(len(y_test))

for i in range(len(y_test)):
    
    if(y_true[i]==1):
        if(y_predict[i]==1):
            TP_list[i] = 1
    else:
        TP_list[i] = 0
        
    if(y_true[i]==0):
        if(y_predict[i]==1):
            FP_list[i] = 1
    else:
        FP_list[i] = 0
        
    if(y_true[i]==0):
        if(y_predict[i]==0):
            TN_list[i] = 0
    else:
        TN_list[i] = 0
        
    
    if(y_true[i]==1):
        if(y_predict[i]==0):
            FN_list[i] = 1
    else:
        FN_list[i] = 0
        
TP = np.sum(TP_list) 
FN =np.sum(FN_list) 
FP = np.sum(FP_list) 
TN = np.sum(TN_list) 
        
    
P = TP /(FP + TP)
R= TP /(TP/FN)

print("F = ", 0.5*(1/P+1/R))
     
    
matrix_TP = np.array([[TP,FN], [FP,TN]])

plt.imshow(matrix_TP)
plt.savefig("matriz_de_confusion.png")
    
