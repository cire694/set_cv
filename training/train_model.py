import torch
import os
import torch 
import json
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



class SetCardDataset(Dataset):

    def __init__(self, root, transform = None):
        #root: where the root of the data is located
        self.root = root
        self.transform = transform

        #image paths and labels should have the same length
        self.image_paths = []
        self.labels = []

        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            
            if not os.path.isdir(folder_path): 
                continue
            
            try:
                attr_label = [int(c) - 1 for c in folder_name] #color, number, shape, fill
            except ValueError:
                continue
            
            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(attr_label)
            
    
    def __len__(self): 
        return len(self.image_paths)


    def __getitem__(self, idx):
        #given an index, returns (image, label)
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: 
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])
            
class ApplyTransform(Dataset): 
    """ Specifies which transform to perform on dataset subset. Train and Test should use different transforms"""

    def __init__(self, subset, transform=None): 
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx): 
        img, label = self.subset[idx]
        if self.transform: 
            img = self.transform(img)
        
        return img, label

    def __len__(self): 
        return len(self.subset)


class SetCardDetector(nn.Module):

    def __init__(self):
        super().__init__()
        #replaced pooling with strides instead; shapes wasn't learning well
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #(3, 150, 200) -> (16, 150, 200)
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 2, padding=1), #(16, 150, 200) -> (32, 75, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 3, padding = 1), #(32, 75, 100) -> (64, 25, 34)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (64, 25, 34) -> (64, 25, 34)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten()#flattens 3d into 1D, (64, 25, 33) -> (64 * 25 * 33,)
        )
        self.color_head = nn.Linear(64*25*34, 3)
        self.number_head = nn.Linear(64*25*34, 3)
        self.shape_head = nn.Linear(64*25*34, 3)
        self.fill_head = nn.Linear(64*25*34, 3)

    
    def forward(self, x):
        raw = self.layers(x)

        return {
            "color": self.color_head(raw),
            "number": self.number_head(raw),
            "shape": self.shape_head(raw),
            "fill": self.fill_head(raw)
        }


        

        


    





def train(model, dataloader, optimizer, loss_fn, device): 

    model.train()
    stats = {"color": 0, "number": 0, "shape": 0, "fill": 0, "total": 0}


    for img, labels in dataloader: #for each batch
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        res = model(img)

        color_loss = loss_fn(res["color"], labels[:, 0]) #slicing
        number_loss = loss_fn(res["number"], labels[:, 1])
        shape_loss = loss_fn(res["shape"], labels[:, 2])
        fill_loss = loss_fn(res["fill"], labels[:, 3])
        
        stats["color"] += color_loss.item()
        stats["number"] += number_loss.item()
        stats["shape"] += shape_loss.item()
        stats["fill"] += fill_loss.item()

        stats["total"] += color_loss.item() + number_loss.item() + shape_loss.item() + fill_loss.item()
            
        total_loss = color_loss + number_loss + shape_loss + fill_loss

        total_loss.backward()
        optimizer.step()


    return stats          


def eval(model, dataloader, loss_fn, device): 
    model.eval()

    with torch.no_grad():
        stats = {"color": 0, "number": 0, "shape": 0, "fill": 0, "total": 0}
  
        fully_correct = 0 #number of cards where all attributes correctly identified
        for img, labels in dataloader: #batched images
            img = img.to(device)
            labels = labels.to(device)

            res = model(img)

            color_loss = loss_fn(res["color"], labels[:, 0]) #slicing
            number_loss = loss_fn(res["number"], labels[:, 1])
            shape_loss = loss_fn(res["shape"], labels[:, 2])
            fill_loss = loss_fn(res["fill"], labels[:, 3])
            
            stats["color"] += color_loss.item()
            stats["number"] += number_loss.item()
            stats["shape"] += shape_loss.item()
            stats["fill"] += fill_loss.item()

            stats["total"] += color_loss.item() + number_loss.item() + shape_loss.item() + fill_loss.item()
            
            #res['color'] returns (B, 3). B is for batch, 3 is for each of the three possibilities. 
            #take the max over the 1st dimension. max returns (value, idx)
            pred_color = torch.max(res["color"], 1)[1]
            pred_number = torch.max(res["number"], 1)[1]
            pred_shape = torch.max(res["shape"], 1)[1]
            pred_fill = torch.max(res["fill"], 1)[1]

            #(B, 4), each pred is (B,). Each value contains an array of len 4 for the prediction
            all_preds = torch.stack([pred_color, pred_number, pred_shape, pred_fill], dim = 1)

            correct_matrix = (all_preds == labels) #matrix of true/false
            perfect_cards = correct_matrix.all(dim = 1) #runs all() on each row. Returns (B,)

            fully_correct += perfect_cards.sum().item() #(1,) to scalar
    return stats , fully_correct / len(dataloader.dataset)



if __name__ == '__main__':

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") #macbook GPU
    else: 
        device = torch.device("cpu")

    print(f"Using device: {device}")

    #set up transformations
    train_transform = transforms.Compose([
        transforms.Resize((150, 200)),
        transforms.RandomRotation(degrees=60), 
        transforms.ColorJitter(brightness=0.3, contrast = 0.2, saturation = 0.2), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale = 0.2, p = 0.3), #distored up to 20%, applied 30% of the time
        # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), #p = 0.2 means happens to 20% of images, scale defines what percent of image area is erased
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(mean), (std). 
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150, 200)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #set up the dataset
    full_dataset = SetCardDataset(root="../data/", transform = None)

    train_size = int(len(full_dataset) * 0.7)
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(full_dataset, [train_size, test_size])

    train_data = ApplyTransform(train_indices, transform=train_transform)
    test_data = ApplyTransform(test_indices, transform=test_transform)
    
    #set up training
    train_loader = DataLoader(
        dataset = train_data, 
        batch_size=128,
        shuffle=True,     #
        num_workers = 8,  #number of subproccesses for data loading
        pin_memory=True,  #faster GPU transfer
        drop_last = False #wether to drop the last batch if it's smaller than batch_size 
    )

    test_loader = DataLoader(
        dataset = test_data, 
        batch_size=128,
        shuffle=False,     #
        num_workers = 8,  #number of subproccesses for data loading
        pin_memory=True,  #faster GPU transfer
        drop_last = False #wether to drop the last batch if it's smaller than batch_size 
    )

    model = SetCardDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    




    
    num_epochs = 50
    best_val_acc = 0.0

    train_losses = []
    eval_losses = []
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        #train
        train_loss_dict = train(model, train_loader, optimizer, loss_fn, device)
        
        eval_loss_dict, accuracy = eval(model, test_loader, loss_fn, device)
        scheduler.step()

        train_loss_dict = {key: val / len(train_loader) for (key, val) in train_loss_dict.items()}
        eval_loss_dict = {key: val / len(test_loader) for (key, val) in eval_loss_dict.items()}

        train_losses.append(train_loss_dict)
        eval_losses.append(eval_loss_dict)
        accuracies.append(accuracy)

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
        tqdm.write(f"Train Loss: {train_loss_dict['total']: .4f} | Eval Acc: {accuracy: .2%}")

        if accuracy > best_val_acc: 
            best_val_acc = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    history = {
        "train_losses": train_losses, 
        "eval_losses": eval_losses, 
        "accuracies": accuracies
    }

    with open("training_history.json", "w") as f: 
        json.dump(history, f)
