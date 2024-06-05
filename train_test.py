import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm.notebook import tqdm, trange

from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from collections import namedtuple

# data directory
file_dir = "/work/aliu10/HE_expression_project/data/Mouse_Brain_Sagittal_Posterior/remove_batch_effects/scale/"

all_data = []
# all_genes = []

# Loop through the file indices
for i in range(2):  
    data_path = os.path.join(file_dir, f"section_{i+1}")   
    with open(data_path, "rb") as fp:  
        data = pickle.load(fp)  
        all_data.append(data)  
    
    # gene_path = os.path.join(file_dir, f"gene_names_section_{i+1}")  
    # with open(gene_path, "rb") as fp:  
    #     genes = pickle.load(fp)  
    #     all_genes.append(genes)     
    
train_set = all_data[0]
test_set = all_data[1]


class HEPatchesDataset(Dataset):
    def __init__(self, image_patches, gene_expressions):
        self.image_patches = image_patches
        self.gene_expressions = gene_expressions
        
        # set transforms
        self.transform_x = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        image = self.image_patches[idx]

#         # Permute the image dimensions if they are not in the format [C, H, W]
#         if image.shape[-1] == 3:  # Assuming the last dimension is channels
#             image = np.transpose(image, (2, 0, 1))  # Change from [H, W, C] to [C, H, W]
            
        image = self.transform_x(image)

        gene_expression = self.gene_expressions[idx]
        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        return image, gene_expression

# train set
he_train = [element[0] for element in train_set]
gene_exp_train = [element[1] for element in train_set]
train_dataset = HEPatchesDataset(he_train, gene_exp_train)
train_size = len(train_set)

# test set
he_test = [element[0] for element in test_set]
gene_exp_test = [element[1] for element in test_set]
test_dataset = HEPatchesDataset(he_test, gene_exp_test)
test_size = len(test_set)


train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_iterator = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, hdim):
        super().__init__()
    
    def forward(self, Q, K, V):
        attention_weights = F.softmax(Q.mm(K.t()) / (K.size(-1) ** 0.5), dim=-1)
        output = attention_weights.mm(V)
        return output

class SelfAttention(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.attention_layer = Attention(hdim)

    def forward(self, y_pred):
        output = self.attention_layer(y_pred, y_pred, y_pred)
        return output

class SA_E(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.sa = SelfAttention(hdim)

    def forward(self, y_pred):
        x = self.sa(y_pred)
        return x   
    
    
class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    
    
class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x
    
    
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])


resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

## start to load pre-trained model
## modified the last layer to the specific number we need

pretrained_model = models.resnet50(pretrained = True)

IN_FEATURES = pretrained_model.fc.in_features 

OUTPUT_DIM = len(test_set[0][1]) # number of genes

fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

pretrained_model.fc = fc

model = ResNet(resnet50_config, OUTPUT_DIM)

model.load_state_dict(pretrained_model.state_dict())

SA_E = SA_E(OUTPUT_DIM)

LR = 1e-4

params = [
          {'params': model.conv1.parameters(), 'lr': LR / 10},
          {'params': model.bn1.parameters(), 'lr': LR / 10},
          {'params': model.layer1.parameters(), 'lr': LR / 8},
          {'params': model.layer2.parameters(), 'lr': LR / 6},
          {'params': model.layer3.parameters(), 'lr': LR / 4},
          {'params': model.layer4.parameters(), 'lr': LR / 2},
          {'params': model.fc.parameters()}
         ]

optimizer = optim.Adam(params, lr=LR) 

model = model.to(device)


def safe_correlation(array1, array2, default_value=0):
    std1 = torch.std(array1)
    std2 = torch.std(array2)

    if std1 == 0 or std2 == 0:
        # One of the arrays has zero variance
        return default_value
    else:
        # Calculate the covariance
        cov = torch.mean((array1 - torch.mean(array1)) * (array2 - torch.mean(array2)))

        # Calculate the correlation coefficient
        corr = cov / (std1 * std2)

        return corr
            
    

num_epochs = 60

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for x, y in tqdm(train_iterator, desc="Training", leave=False, disable=True):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(x)
        y_pred_new = SA_E(y_pred)
        loss = torch.mean((y - y_pred_new) ** 2) # torch.mean((y - y_pred) ** 2) + 
        # loss = bleep_loss(y, y_pred_new)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_iterator)
    
    
    # Evaluation
    y_pred_tests = []
    y_tests = []

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in tqdm(test_iterator, desc="Evaluating", leave=False, disable=True):
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)
            y_pred_new = SA_E(y_pred)
            loss = torch.mean((y - y_pred_new) ** 2) # torch.mean((y - y_pred) ** 2) + 
            # loss = bleep_loss(y, y_pred_new)
            test_loss += loss.item()

            y_pred_tests.append(y_pred_new.cpu())
            y_tests.append(y.cpu())
    
        test_loss = test_loss / len(test_iterator)
    
    # Concatenate y_pred_test and y_test
    y_pred_test = torch.cat(y_pred_tests, dim=0)
    y_test = torch.cat(y_tests, dim=0)

    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
    
    ####
    cor_test = []
    
    for i in range(len(test_set[0][1])):
        corr_test = safe_correlation(y_pred_test.transpose(0,1)[i], y_test.transpose(0,1)[i])
        cor_test.append(corr_test)
    print(f'{epoch+1} | \tMean correlation: {sum(cor_test)/len(cor_test):.4f}')
    
    # Select the top 50 most highly expressed genes in y_pred_test
    mean_expression = torch.mean(y_pred_test, axis=0)
    top_50_indices = torch.argsort(mean_expression, descending=True)[:50]
    y_pred_top_50 = y_pred_test[:, top_50_indices]
    y_test_top_50 = y_test[:, top_50_indices]
    
    cor_test_top50 = []
   
    for i in range(50):
        corr_test_top50 = safe_correlation(y_pred_top_50.transpose(0,1)[i], y_test_top_50.transpose(0,1)[i])
        cor_test_top50.append(corr_test_top50)
    print(f'{epoch+1} | \tTop 50 HEG correlation: {sum(cor_test_top50)/len(cor_test_top50):.4f}')
    
    
    path_dir = '/work/aliu10/HE_expression_project/results/Mouse_Brain_Sagittal_Posterior/resnet_s2/'

    # Pair each gene name with its corresponding correlation value
    paired_list = [(gene, cor) for gene, cor in zip(gene_names, cor_test)]
    # Sort the list based on the second element of each tuple
    sorted_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
    save_path = f'{path_dir}sorted_correlations_epoch_{epoch+1}.pt'
    torch.save(sorted_list, save_path)

    formatted_gene_correlations = ', '.join([f'{gene}: {val:.4f}' for gene, val in sorted_list[:5]])
    print(f'{epoch+1} | \tTop 5 correlations: {formatted_gene_correlations}')

    # Save y_pred_test and y_test
    torch.save(y_pred_test, f'{path_dir}y_pred_test_epoch_{epoch+1}.pt')
    torch.save(y_test, f'{path_dir}y_test_epoch_{epoch+1}.pt')
