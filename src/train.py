from torch import nn 
from torch import optim
from torch.cuda.memory import reset_accumulated_memory_stats 
from model import ResNet10
from data import bird_train_dataloader,bird_valid_dataloader
from data import mnist_train_dataloader, mnist_valid_dataloader
import torch
from config import CONFIG
from tqdm import tqdm
import wandb

def validation_loss_accuracy(net):
    print('Validating')
    correct = 0
    total = 0
    total_loss = 0
    total_iterations = 0
    net = net.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(bird_valid_dataloader):
            images, labels = data
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_iterations += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(total)
    validation_accuracy = (100* (correct/total))
    validation_loss = total_loss/total_iterations
    if validation_accuracy > CONFIG['best_validation_accuracy']:
        torch.save(net,'../model_weights/best.pth')
        CONFIG['best_validation_accuracy'] = validation_accuracy
    print(f'Validation Accuracy: {validation_accuracy} Validation Loss: {validation_loss}')
    wandb.log({'Validation Accuracy':validation_accuracy,'Validation Loss': validation_loss})




def training():

    epochs = CONFIG['epochs']
    net = ResNet10().to(CONFIG['device'])

    if CONFIG['use_pretrained']:
        net = torch.load(CONFIG['pretrained_model_path'])
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=CONFIG['learning_rate'], momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])

    
    net = net.train()
    print('Training Started')
    for epoch in range(epochs):
        running_loss = 0.0
        total_iterations = 0
        for data in tqdm(bird_train_dataloader):
            images, labels = data
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            optimiser.zero_grad()
            
            outputs = net(images)
            
            loss = criterion(outputs,labels)
            loss.backward()
            
            optimiser.step()
            
            running_loss += loss.item()
            total_iterations += 1
            #break
            
        torch.save(net,'../model_weights/last.pth')
        if epoch % CONFIG['save_weights_epoch'] == 0:
            torch.save(net,f'../model_weights/{epoch}_bird_clf.pth') 
        print(f'Epoch: {epoch+1} loss: {running_loss/total_iterations}')
        wandb.log({'Training_loss':running_loss/total_iterations})
        validation_loss_accuracy(net)
        net.train()
                
    print('Finished Training!!')