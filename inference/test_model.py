import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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


        
#loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SetCardDetector().to(device)
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
model.eval()

labels_map = {
    "color": ["Red", "Green", "Purple"],
    "number": ["1", "2", "3"], 
    "shape": ["Diamond", "Squiggle", "Oval"], 
    "fill": ["Empty", "Striped", "Solid"] #very strange, order should be swapped
}

def predict(image_path): 
    transform = transforms.Compose([
        transforms.Resize((150, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) #add batch dimension

    with torch.no_grad(): 
        outputs = model(img_tensor)
    
    #output returns a dictionary mapping attribute to a 4x1 logit. 
    results = {}
    for attr, logits in outputs.items(): 
        idx = torch.argmax(logits, dim=1).item()
        results[attr] = labels_map[attr][idx]
    
    return results

    