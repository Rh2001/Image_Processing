import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification

# Initial check
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise fallback to CPU(takes too long)
print("Using device:", device)

# CNN transforms
transform_train = transforms.Compose([
    transforms.Resize((32, 32)), # Ensures all images are the same size for CNN input, even though CIFAR-10 is already 32x32
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Datasets and loaders
trainset = torchvision.datasets.CIFAR10(
    root='./img_data',
    train=True,
    transform=transform_train,
    download=True,
)

testset = torchvision.datasets.CIFAR10(
    root='./img_data',
    train=False,
    transform=transform_test,
    download=True,
)

transform_resnet_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  
])

transform_resnet_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
])

trainset_resnet = torchvision.datasets.CIFAR10(
    root='./img_data',
    train=True,
    transform=transform_resnet_train,
    download=True,
)

testset_resnet = torchvision.datasets.CIFAR10(
    root='./img_data',
    train=False,
    transform=transform_resnet_test,
    download=True,
)

trainloader_resnet = DataLoader(trainset_resnet, batch_size=32, shuffle=True)
testloader_resnet = DataLoader(testset_resnet, batch_size=32, shuffle=False)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

print("Number of training images:", len(trainloader.dataset))
print("Number of test images:", len(testloader.dataset))

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

cnn_model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("\nTraining CNN...")
for epoch in range(10):
    cnn_model.train()
    epoch_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"CNN Epoch {epoch+1}, Loss: {epoch_loss/len(trainloader):.4f}")

cnn_model.eval()
cnn_preds = []
cnn_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)

        cnn_preds.extend(predicted.cpu().numpy())
        cnn_labels.extend(labels.numpy())

cnn_accuracy = accuracy_score(cnn_labels, cnn_preds)
cnn_precision = precision_score(cnn_labels, cnn_preds, average='macro')
cnn_recall = recall_score(cnn_labels, cnn_preds, average='macro')
cnn_f1 = f1_score(cnn_labels, cnn_preds, average='macro')

print("\nCNN Metrics:")
print(f"Accuracy : {cnn_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall   : {cnn_recall:.4f}")
print(f"F1-Score : {cnn_f1:.4f}")

print("\nCNN Classification Report:") # Check the 10 classes and their performance
print(classification_report(cnn_labels, cnn_preds))

# Fine-tuning ResNet
print("\nFine-tuning ResNet as backbone with hyperparameter testing...")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

learning_rates = [0.001, 0.0001]
dropout_rates = [0.3, 0.5]

results = {}

for lr in learning_rates:
    for dropout_rate in dropout_rates:
        print(f"Training ResNet with lr={lr}, dropout={dropout_rate}")

        resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        for name, param in resnet_model.resnet.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False

        in_features = 2048

        resnet_model.classifier = nn.Sequential(
            nn.Flatten(), # Stops attribute error for ResNetForImageClassification, my program would crash otherwise
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )

        resnet_model.config.num_labels = 10
        resnet_model = resnet_model.to(device)

        resnet_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, resnet_model.parameters()),
            lr=lr
        )

        for epoch in range(5):
            resnet_model.train()
            epoch_loss = 0.0

            for images, labels in trainloader_resnet:
                labels = labels.to(device)

                # Convert tensors to PIL images
                pil_images = [transforms.ToPILImage()(img) for img in images]
                pixel_values = image_processor(images=pil_images, return_tensors="pt").pixel_values.to(device)

                resnet_optimizer.zero_grad()
                outputs = resnet_model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                resnet_optimizer.step()

                epoch_loss += loss.item()

            print(f"ResNet Epoch {epoch+1}, Loss: {epoch_loss/len(trainloader_resnet):.4f}") # Calculate average loss for the epoch using loss/total number

        resnet_model.eval()
        resnet_preds = []
        resnet_labels = []

        with torch.no_grad():
            for images, labels in testloader_resnet:
                labels = labels.to(device)
                pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
                pixel_values = image_processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
                outputs = resnet_model(pixel_values=pixel_values)
                preds = torch.argmax(outputs.logits, dim=-1)

                resnet_preds.extend(preds.cpu().numpy())
                resnet_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(resnet_labels, resnet_preds)
        precision = precision_score(resnet_labels, resnet_preds, average='macro')
        recall = recall_score(resnet_labels, resnet_preds, average='macro')
        f1 = f1_score(resnet_labels, resnet_preds, average='macro')

        # Metrics up to 4 decimal.
        print("\nResNet Metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

        results[(lr, dropout_rate)] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

print("Hyperparameter Testing Completed")

best_params = max(results, key=lambda x: results[x]["f1"]) # Return best F1 score parameters
best_metrics = results[best_params]

print(f"\nBest Hyperparameters:")
print(f"Learning Rate = {best_params[0]}")
print(f"Dropout       = {best_params[1]}")
print("\nBest Model Metrics:")
print(f"Accuracy : {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall   : {best_metrics['recall']:.4f}")
print(f"F1-Score : {best_metrics['f1']:.4f}")
