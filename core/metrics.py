import torch

def compute_iou(pred,mask,num_classes):
    pred=torch.argmax(pred,1)
    ious=[]
    for cls in range(num_classes):
        p=pred==cls
        t=mask==cls
        inter=(p&t).sum().float()
        union=(p|t).sum().float()
        if union==0: continue
        ious.append((inter/union).item())
    return sum(ious)/len(ious) if len(ious)>0 else 0