import torch
from config import CONFIG
from data import bird_test_dataloader
from torch import nn
from tqdm import tqdm



def testing(net):
    print('Testing')
    correct = 0
    total = 0
    total_loss = 0
    total_iterations = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(bird_test_dataloader):
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
            
    print(f'Test Accuracy: {(100* (correct/total))} Test Loss: {total_loss/total_iterations}')





