import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tensorflow as tf
import os
from torch.optim import AdamW
import random
from Vision_Transformer import ViT 
import copy
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

#%% Set GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #close info+warning
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #close info
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config) 

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#%% Read file
path = r''
train_data = np.load(path+'\\')
train_gt = np.load(path+'\\')
train_sub = np.load(path+'\\')

test_data = np.load(path+'\\')
test_gt = np.load(path+'\\')
test_sub = np.load(path+'\\')

#%% Show confusion matrix
def show_CM(validation,prediction):
    matrix = metrics.confusion_matrix(validation,prediction)
    plt.figure(figsize = (6,4))
    sns.heatmap(matrix,cmap = 'coolwarm',linecolor= 'white',
                linewidths= 1,annot= True,fmt = 'd')
    plt.title('Pretrain Teacher')
    plt.ylabel('True')
    plt.xlabel('Prediction')
    plt.plot()
    plt.show()
    
#%% Random seed
def seed_everything(seed=24):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed 
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
seed_everything(113)

#%% Focal loss
class focal_loss(nn.Module):
    def __init__(self, alpha=[0.25,0.75], gamma=2, num_classes=2, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha: α,class weight. retainnet set as 0.25
        :param gamma: γ,degree parm. retainnet set as 2
        :param num_classes: class num 
        :param size_average: loss cal
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            # If α is a list, size:[num_classes], [a, b]
            assert len(alpha) == num_classes

            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1

            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            # α = [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
     
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) #nll_loss(cross entropy = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) # torch.pow((1-preds_softmax), self.gamma) = focal loss (1-pt)**γ
 
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss  
        
#%% Cosine learning rate schedule with warmup
def get_cosine_schedule_with_warmup(
  optimizer: Optimizer, # The optimizer for which to schedule the learning rate.
  num_warmup_steps: int, # The number of steps for the warmup phase.
  num_training_steps: int, # The total number of training steps.
  num_cycles: float = 0.5, # The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).
  last_epoch: int = -1, # The index of the last epoch when resuming training.
):

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)   
  
#%% Testing process
def evaluate(model, device, criterion, test_loader):
    with torch.no_grad():
        val_preds = []
        val_gt = []
        val_losses = []
        model.eval() 
        for i, (x, y) in enumerate(test_loader):
            inputs ,targets = x.to(device) ,y.to(device)
            inputs = inputs.to(torch.float32)
            val_output = model(inputs)
           
            val_loss = criterion(val_output, targets.long())
            val_losses.append(val_loss.item())
            val_output = F.log_softmax(val_output, dim =-1)
        # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(val_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().detach().numpy().flatten()
            val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
            val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))
            del y_preds, y_true      
    
        ori_preds = copy.deepcopy(val_preds)   
        for i in range(len(val_preds)):
           if val_preds[i] == 1 and val_gt[i] == 1: 
               for j in range(-2,3):
                   if  i+j < len(val_preds) and val_gt[i+j] == 1:
                       val_preds[i+j] = 1
            
        print("Val_Loss: {:5.4f}".format(np.mean(val_losses)),
            "Val Acc: {:.5f}".format(accuracy_score(val_gt, val_preds)),
            "Val Prec: {:.5f}".format(precision_score(val_gt, val_preds)),
            "Val Rcll: {:.5f}".format(recall_score(val_gt, val_preds)),
            "Val F1: {:.5f}".format(f1_score(val_gt, val_preds)))       
    
    return val_preds, val_gt, val_losses, ori_preds    
    
#%% Training process
config1 = {
    'epochs': 150,
    'batch_size': 64,
    'learning_rate': 1e-3
}

train_data = np.expand_dims(train_data, axis = 1)
train_data = np.transpose(train_data,(0,1,3,2))
test_data = np.expand_dims(test_data, axis = 1)
test_data = np.transpose(test_data,(0,1,3,2))
x_train_tensor = torch.from_numpy(train_data).to(device)
x_test_tensor = torch.from_numpy(test_data).to(device)
y_train_gd_tensor =  torch.from_numpy(train_gt).to(device)
y_test_gd_tensor =  torch.from_numpy(test_gt).to(device) 

model = ViT(
        image_size = 50,
        patch_height = 3,
        patch_width = 10,
        num_classes = 2,
        channels=1,
        dim = 64,
        depth = 3,
        heads = 3,
        mlp_dim = 128,
        dropout = 0.2,
        dim_head = 64,
        emb_dropout = 0.2
    ).to(device)

optimizer = AdamW(model.parameters(),lr=config1['learning_rate'])
criterion = focal_loss().to(device)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps= config1['epochs'])

deal_dataset = TensorDataset(x_train_tensor, y_train_gd_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_gd_tensor)
trainloader = DataLoader(deal_dataset, batch_size=config1['batch_size'],shuffle=True,drop_last=True)
testloader = DataLoader(test_dataset, batch_size=config1['batch_size'],shuffle=False)  

loss_test = []
loss_train = []
model.train()
for epoch in range(config1['epochs']):
    train_losses = []
    train_preds = []
    train_gt = []
    start_time = time.time()
    for i, (x,y) in enumerate(trainloader):
        inputs, labels = x.to(device), y.to(device)
        inputs = inputs.to(torch.float32)
        
        optimizer.zero_grad()
         
        pred = model(inputs)
        loss = criterion(pred, labels.long())
        loss.backward() 
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)                                                                                                                         
        optimizer.step()
        
        train_output = F.log_softmax(pred, dim =-1)
            
        train_losses.append(loss.item())
        y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
        y_true = labels.cpu().detach().numpy().flatten()
        train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
        train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))
        del y_preds, y_true
    cur_loss = np.mean(train_losses)
    elapsed = time.time() - start_time
    print('| epoch {:3d} | {:5.4f} s/epoch | train loss {:5.4f} | train accuracy {:5.4f}'.format(epoch, elapsed, cur_loss, (accuracy_score(train_gt, train_preds))))
    start_time = time.time()
    loss_train.append(np.mean(train_losses))
    
    val_preds, val_gt, val_losses, ori_preds = evaluate(model, device, criterion, testloader)
    scheduler.step()
    loss_test.append(np.mean(val_losses))
    model.train()
    
plt.plot(loss_train, label = 'trian_loss')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
plt.plot(loss_test, label = 'test_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["Train Loss","Test Loss"],loc = 'upper right')
plt.show()    
plt.clf()

print("Train Acc: {:.5f}".format(accuracy_score(train_gt, train_preds)),
    "Train Prec: {:.5f}".format(precision_score(train_gt, train_preds)),
    "Train Rcll: {:.5f}".format(recall_score(train_gt, train_preds)),
    "Train F1: {:.5f}".format(f1_score(train_gt, train_preds)),
    "Val Acc: {:.5f}".format(accuracy_score(val_gt, val_preds)),
    "Val Prec: {:.5f}".format(precision_score(val_gt, val_preds)),
    "Val Rcll: {:.5f}".format(recall_score(val_gt, val_preds)),
    "Val F1: {:.5f}".format(f1_score(val_gt, val_preds)))
show_CM(val_gt, val_preds)

torch.save(model, 'teacher.pkl')

#%% Lead time
lead = []
for i in range(len(ori_preds)):
    if (ori_preds[i] == val_gt[i] == 1): 
        if (val_gt[i-1] == 0):
            lead.append(10)
        elif (val_gt[i-1] == 1) and (val_gt[i-2] == 0) and (ori_preds[i-1] == 0) :
            lead.append(20)
        elif (val_gt[i-1] == val_gt[i-2] == 1) and (ori_preds[i-1] == ori_preds[i-2] == 0):
            lead.append(30)
           
lead = np.array(lead)            
print(lead.mean())
print(lead.std())

#%% Flops
from ptflops import get_model_complexity_info
flops1, params1 = get_model_complexity_info(model, (50,3), as_strings=False, print_per_layer_stat=True)
print('flops: ', flops1, 'params: ', params1)    
