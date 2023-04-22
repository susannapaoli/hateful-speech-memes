import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict



def train(model, data_loader, test_loader, criterion, optimizer, lr_scheduler, modelpath, writer, device, epochs):
    train_loss= []
    v_loss = []
    v_acc = []

    model.train()

    for epoch in range(epochs):
        avg_loss = 0.0


        for batch_num, (captions, input_id, attention_masks, target) in enumerate(data_loader):
            

            target = target.to(device)
            input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
                
            
            outputs = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_masks, 
                             labels=target
                                )
            loss = outputs.loss
                   
            '''
            Take Step
            '''                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            '''
            Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            
            avg_loss += loss.item()


            '''
            linear_schedule_with_warmup take step after each batch
            '''
            lr_scheduler.step()
            
                
            
            del captions
            del input_ids
            del attention_masks
            del target
            del loss
            
            
        training_loss = avg_loss/len(data_loader)
        
       
        print('Epoch: ', epoch+1)            
        print('training loss = ', training_loss)
        train_loss.append(training_loss)

        valid_loss, acc= validation(model, test_loader, criterion, device)
        print('Validation Loss: {:.4f}\tValidation Accuracy: {:.4f}'.format(valid_loss, acc))
        v_loss.append(valid_loss)
        v_acc.append(acc)

    
        writer.add_scalar("Loss/train", training_loss, epoch)            
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/Validation', acc, epoch)
        return train_loss, v_loss, v_acc

       
      

        

def validation(model, test_loader, criterion, device):
    model.eval()
    test_loss = []
    correct = 0
    total = 0

    with torch.no_grad():
      for batch_num, (captions, input_id, attention_masks, target) in enumerate(test_loader):
          
          target = target.to(device)
          input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
              
          
          outputs = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_masks, 
                             labels=target
                                )
          loss = outputs.loss
          test_loss.append(loss.item())
          predicted = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
          correct += (predicted == target).sum().item()
          total += target.size(0)

          
          
          del captions
          del input_ids
          del attention_masks
          del target
          del loss
              
      model.train()
      return np.mean(test_loss), correct/total
