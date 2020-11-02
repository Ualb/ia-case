
class Trainer:

    def evaluate(model, dataset_loader, optimizer, train=False):
    #Contador de aciertos y acumulador de la función de pérdida
    correct_cnt, ave_loss = 0, 0 
    #Contador de muestras
    count = 0 

    for batch_idx, (x, target) in enumerate(dataset_loader):
        #sumamos el tamaño de batch, esto es porque n_batches*tamaño_batch != n_muestras
        count += len(x) 
        if train:
            #iniciamos a 0 los valores de los gradiente
            optimizer.zero_grad() 

        #Convertimos el tensor a variable del modulo autograd
        x, target = Variable(x), Variable(target) 
        #realizamos el forward
        score, loss = model(x, target) 
        #pasamos de one hot a número
        _, pred_label = torch.max(score.data, 1) 
        #calculamos el número de etiquetas correctas
        correct_cnt += (pred_label == target.data).sum() 
        #sumamos el resultado de la función de pérdida para mostrar después
        ave_loss += loss.data[0] 
        if train:
            #Calcula los gradientes y los propaga
            loss.backward()  
            #adaptamos los pesos con los gradientes propagados
            optimizer.step() 

    #Calculamos la precisión total
    accuracy = correct_cnt/count 
    #Calculamos la pérdida media
    ave_loss /= count 
    #Mostramos resultados
    print ('==>>>loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy)) 
