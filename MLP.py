class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256) # capa oculta
        self.fc2 = nn.Linear(256, 10) # capa de salida
        self.loss_criterion = nn.CrossEntropyLoss() # Función de pérdida
        
    # operaciones desde su entrada hasta su salida
    def forward(self, x, target):
        x = x.view(-1, 28*28) # transforma las imágenes de tamaño (n, 28, 28) a (n, 784)
        x = F.relu(self.fc1(x)) # Función de activación relu en la salida de la capa oculta
        x = F.softmax(self.fc2(x), dim=1) # Función de activación softmax en la salida de la capa oculta
        loss = self.loss_criterion(x, target) # Calculo de la función de pérdida
        return x, loss