from audioop import mul
import torch
import torch.utils.data as data
from Model.Network import Network
from Dataloader import DatasetSegmentation, transformSeg
import numpy as np
from Metrix import HarmonicMean, CM, Weight
import torch.optim as optimizers

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def EarlyStopping(valid_loss,model,PATH):
    if(valid_loss < EarlyStopping.best_loss): 
         print("Loss has reduced so saving the model")
         torch.save(model.state_dict(), PATH)
         EarlyStopping.best_loss = valid_loss
         EarlyStopping.count = 0
    else:
        EarlyStopping.count += 1

    if(EarlyStopping.count > 50):
        EarlyStopping.stop = True


def Step(model, batch, opt, loss_fun):
    img, label = batch
    preds = model(img)
    loss = loss_fun(preds,label)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss



def train_model(model, batch, mult, n_epochs, opt, PATH, w, device):

    # Initialize the early stopping object
    EarlyStopping.stop = False
    EarlyStopping.count = 0
    EarlyStopping.best_loss = 20

    # Initialize the dataloaders
    Dataset = DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=transformSeg)
    l = len(Dataset)
    trainDataset , valDataset= torch.utils.data.random_split(Dataset,[l-100,100],generator=torch.Generator().manual_seed(42))
    train_dl = torch.utils.data.DataLoader(trainDataset,batch_size = batch,shuffle = True)
    val_dl = torch.utils.data.DataLoader(valDataset,batch_size = 10,shuffle = True)

    print("weight: ",w)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    # Initialize all the metric we would use
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    TPs = []
    FPs = []
    TNs = []
    FNs = []
    Unions = []
    
    # start training
    for epoch in range(1, n_epochs + 1):

        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_dl, 1):
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            loss = loss / mult
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            if(batch % mult == 0):
                opt.step()
                opt.zero_grad()
            # record training loss
            train_losses.append(loss.item())

        model.eval() 
        # prep model for evaluation
        TPs = []
        FPs = []
        TNs = []
        FNs = []
        Unions = []
        # Initialize the list for different matrics
        for data, target in val_dl:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            # calculate the Confusion Matrixs and Union
            TP,FP,TN,FN,union = CM(output,target)

            # record validation loss
            valid_losses.append(loss.item())
            # Record the confusion matrix and Union
            TPs.append(TP)
            FPs.append(FP)
            TNs.append(TN)
            FNs.append(FN)
            Unions.append(union)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # Element wise sum the confusion matrix
        mTP = torch.sum(torch.stack(TPs),dim=0)
        mFP = torch.sum(torch.stack(FPs),dim=0)
        mTN = torch.sum(torch.stack(TNs),dim=0)
        mFN = torch.sum(torch.stack(FNs),dim=0)
        mUnion = torch.sum(torch.stack(Unions), dim = 0)

        # Calculate Precision Recall and F_score
        Precision = torch.div(mTP,torch.add(mTP , mFP))
        Recall  = torch.div(mTP,torch.add(mTP , mFN))
        mF_score = HarmonicMean(Precision,Recall)
        mIOU = torch.div(mTP,mUnion)


        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        print("mIOU: ",mIOU)
        print("mPrecision: ",Precision)
        print("mRecall: ",Recall)
        print("mF_score: ",mF_score)

        print("FP: ", mFP ," TP: ", mTP, " FN: ", mFN, " TN: ", mTN)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        EarlyStopping(valid_loss, model,PATH)
        
        if EarlyStopping.stop:
            print("Early stopping")
            break

    return  avg_train_losses, avg_valid_losses

if __name__ == "__main__":
    model = Network(256,8)
#    train_model(model,2,1,optimizers.Adam(model.parameters()),"./logs/Train_test.pt","cpu")
    