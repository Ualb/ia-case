
# importaciones
import torch
# La librería de redes neuronales que utilizaremos para crear nuestro modelo.
import torch.nn as nn 
import torch.nn.functional as F
#  En concreto el módulo Variable de esta librería que se encarga de manejar 
# las operaciones de los tensores.
from torch.autograd import Variable
# El módulo que ayudará a cargar el conjunto de datos que vamos a utilizar 
# y explicaremos más adelante.
import torchvision.datasets as dset
# Este módulo contiene una serie de funciones que nos ayudarán modificando el dataset.
import torchvision.transforms as transforms
# De aquí usaremos el optimizador para entrenar la red neuronal y modificar sus pesos.
import torch.optim as optim


# clase local
import MLP
from trainer import Trainer

"""
Una vez importados los módulos, debemos cargar el conjunto
de datos que vamos a utilizar, en este tutorial vamos a utilizar
el conjunto conocido como MNIST. Se trata de un conjunto de imágenes
de dígitos escritos a mano, contiene un total de 60000
imágenes para entrenamiento y 10000 para test. Todos los dígitos
están normalizados en tamaño y centrados en la imagen de tamaño
28x28 en escala de grises. El objetivo de esta base de datos es 
clasificar cada imagen diciendo a que número entre el 0 y el 9 pertenece.
"""

# este dataset ya viene normalizado en valores de 0 - 1
torch.manual_seed(123) #fijamos la semilla
trans = transforms.Compose([transforms.ToTensor()]) #Transformador para el dataset


root="./data" # ruta en donde deseamos descargar los datos
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)

# tamaños de conjuntos de datos que vamos a pasar a la red
batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# imprimiendo la cantidad de batches
print ('Trainning batch number: {}'.format(len(train_loader)))
print ('Testing batch number: {}'.format(len(test_loader)))

# **********************************************************
#                   Creacion del modelo de
#                       Entrenamiento
# **********************************************************

# topología de una capa con 256 neuronas
# representada en la clase MLP
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# **********************************************************
#                   Entrenamiento del modelo 
# **********************************************************

coach = Trainer()
for epoch in range(10):
    print("Epoch: {}".format(epoch))
    print("Train")
    coach.evaluate(model, train_loader, optimizer, train=True)
    print("Test")
    coach.evaluate(model, test_loader, optimizer, train=False)

# guardar y cargar el modelo creado
torch.save(model, "nombre_del_modelo")
#Carga el modelo
model = torch.load("nombre_del_modelo")

torch.save(model.state_dict(),"nombre_del_modelo")
#Generamos un nuevo modelo
model = MLP()
#Cargamos los pesos anteriores
model.load_state_dict(torch.load("nombre_del_modelo"))
