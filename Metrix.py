import torch

def HarmonicMean(x,y):
    return torch.div(torch.mul(2*x,y) , torch.add(x,y))

def CM(y,label):
    predicted_label = torch.argmax(y,dim = 1)
    TP = torch.stack([torch.logical_and(label == i,predicted_label == i).sum() for i in range(8)])
    FP = torch.stack([torch.logical_and(label != i,predicted_label == i).sum() for i in range(8)])
    TN = torch.stack([torch.logical_and(label != i,predicted_label != i).sum() for i in range(8)])
    FN = torch.stack([torch.logical_and(label == i,predicted_label != i).sum() for i in range(8)])
    union = torch.stack([torch.logical_or(label == i,predicted_label == i).sum() for i in range(8)])
    return TP,FP,TN,FN,union

def Weight(label):
    W = torch.stack([(label == i).sum() for i in range(8)])
    return W / (256*256)

   