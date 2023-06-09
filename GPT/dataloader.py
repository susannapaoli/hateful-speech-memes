import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils import data
from torchvision import transforms, datasets, models
from PIL import Image
import json

'''
Load the GPT tokenizer.
'''
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
#pad_token = '[PAD]'
#pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
#tokenizer.pad_token = pad_token
#tokenizer.pad_token_id = pad_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id




'''
Dataloader for Training/Validation
Returns (Image, Caption, Input_id, Attention_mask, label)
'''
class mydataset():    

    def __init__(self, classification_list, name):

        super(mydataset).__init__()
        
        self.X = []
        self.Cap = []
        self.Y = []
        
        with open(classification_list, mode = 'r') as f:
            
            for line in f:
                #path, caption, label = line[:-1].split('\t')
                data = json.loads(line)
                
                path = data['img']
                caption = data['text']
                label = data['label']
                self.X.append('/content/data/'+path)
                self.Cap.append(caption)
                self.Y.append(label)
                
        
        '''
        Tokenize all of the captions and map the tokens to thier word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.Cap)
        
        
        
        '''
        Image Transforms
        '''
        
        if name in ['valid','test']:
            self.transform = transforms.Compose([ transforms.Resize(384),
                                                 transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([ transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                                                            ])
    
    
    def __getitem__(self,index):
        '''
        For Image and Label
        '''
        #image = self.X[index]
        #image = Image.open(image).convert('RGB')

        #image = self.transform(image)

        label = float(self.Y[index])
        '''
        For Captions, Input ids and Attention mask
        '''
        caption = self.Cap[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        
        return caption, input_id, attention_masks, torch.as_tensor(label).long()
        
  
    def __len__(self):
        return len(self.X)
    
    
    
    
    
    
    
        
'''
tokenize all of the sentences and map the tokens to their word IDs.
'''

def tokenize(sequences):
    
    input_ids = []
    attention_masks = []

    # For every caption...
    for seq in sequences:
        '''
        `encode_plus` will:
          (1) Tokenize the caption.
          (2) Prepend the `[CLS]` token to the start.
          (3) Append the `[SEP]` token to the end.
          (4) Map tokens to their IDs.
          (5) Pad or truncate the sentence to `max_length`
          (6) Create attention masks for [PAD] tokens.
        '''
        encoded_dict = tokenizer.encode_plus(
                            seq,                       # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 48,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            #padding=True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',      # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    
    return input_ids, attention_masks

'''
Dataloader for Training/Validation
Returns (Image, Caption, Input_id, Attention_mask, label)
'''
class myfusiondataset():    

    def __init__(self, classification_list, name):

        super(mydataset).__init__()
        
        self.X = []
        self.Cap = []
        self.Y = []
        
        with open(classification_list, mode = 'r') as f:
            
            for line in f:
                #path, caption, label = line[:-1].split('\t')
                data = json.loads(line)
                
                path = data['img']
                caption = data['text']
                label = data['label']
                self.X.append('/content/data/'+path)
                self.Cap.append(caption)
                self.Y.append(label)
                
        
        '''
        Tokenize all of the captions and map the tokens to thier word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.Cap)
        
        
        
        '''
        Image Transforms
        '''
        
        if name in ['valid','test']:
            self.transform = transforms.Compose([ transforms.Resize(384),
                                                 transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([ transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                                                            ])
    
    
    def __getitem__(self,index):
        '''
        For Image and Label
        '''
        image = self.X[index]
        image = Image.open(image).convert('RGB')

        image = self.transform(image)

        label = float(self.Y[index])
        '''
        For Captions, Input ids and Attention mask
        '''
        caption = self.Cap[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        
        return image, caption, input_id, attention_masks, torch.as_tensor(label).long()
        
  
    def __len__(self):
        return len(self.X)