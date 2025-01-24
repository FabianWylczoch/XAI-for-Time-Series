import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import sklearn

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F



class UCRDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label


class DataHandler():
    def __init__(self, train_x, test_x, train_y, test_y):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

    def cancel_time_series(self, ts, start, end, data, how="mean"):
        out = ts.copy()
        if how == "mean":
            out[start:end] = np.array([np.mean([ts[i] for ts in data]) for i in range(start, end)]).reshape(end - start,1)
        if how == "linear":
            out[start:end] = np.interp(list(range(start, end)), [start, end], [out[start][0], out[end-1][0]]).reshape(end - start,1)
        if how == "zero":
            out[start:end] = 0
        return out    
    
    def getData(self, how="raw", all_areas=None, invert=False):
        if how == "raw":
            return self.train_x, self.test_x, self.train_y, self.test_y 
        if invert:
            all_inverted = []
            for dataset_areas in all_areas:
                dataset_inverted = []
                for sample_areas in dataset_areas:
                    sample_inverted = []
                    previous_end = 0
                    for start, end in sample_areas:
                        if start > previous_end:
                            sample_inverted.append([previous_end, start])
                        previous_end = end
                    if previous_end < self.train_x.shape[1]:
                        sample_inverted.append([previous_end, self.train_x.shape[1]])
                    dataset_inverted.append(sample_inverted)
                all_inverted.append(dataset_inverted)
            all_areas = all_inverted

        data = [self.train_x, self.test_x]
        out_data = []
        for i_ds, dataset_areas in enumerate(all_areas):
            dataset_x = []
            for i_sample, sample_areas in enumerate(dataset_areas):
                sample_x = data[i_ds][i_sample]
                for start, end in sample_areas:
                    sample_x = self.cancel_time_series(sample_x, start, end, data[i_ds], how=how)
                dataset_x.append(sample_x)
            out_data.append(np.array(dataset_x))
        out_data.append(self.train_y)
        out_data.append(self.test_y)
        return out_data


class BasicCNN(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce size by half
        self.fc1 = nn.Linear(64 * int(input_shape/4), 128)  
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        self.features = x  # Save the features before flattening for Grad-CAM
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    

class ClassificationModel():
    def __init__(self, train_x, train_y):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = train_x.shape[1]
        self.model = BasicCNN(input_shape = self.input_shape).to(self.device)
        self.enc = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(train_y.reshape(-1,1))


    def createDataLoader(self, X, y):
        X = X.reshape(-1, 1, X.shape[-2])
        dataset = UCRDataset(X.astype(np.float32),y.astype(np.float32))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        return dataloader    


    def fit(self, X, y, num_epochs = 20):
        
        y = self.enc.transform(y.reshape(-1,1))

        loader = self.createDataLoader(X, y)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(loader):.4f}")


    def test(self, X, y):
        y = self.enc.transform(y.reshape(-1,1))
        loader = self.createDataLoader(X, y)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += 1
                correct += (predicted == torch.max(labels.reshape(1, -1), 1)[1]).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")


    def getConfidences(self, X):
        X = X.reshape(-1, 1, X.shape[-2])
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
            outputs = self.model(X)
        return outputs


    def classify(self, index, X, y):
        y = self.enc.transform(y.reshape(-1,1))
        self.model.eval()
        sample_x = X[index].reshape(-1, 1, X.shape[-2])
        sample = UCRDataset(sample_x.astype(np.float32), y[index].astype(np.float32))
        with torch.no_grad():
            image = torch.from_numpy(sample.feature).to(self.device) # Add batch dimension
            output = self.model(image)

            predicted = np.zeros(len(sample.target))
            _, ind = torch.max(output, 1)
            predicted[ind] = 1.0
        return np.argmax(predicted), np.argmax(sample.target)
    

    def grad_cam(self, index, X, y, target_class):
        y = self.enc.transform(y.reshape(-1,1))
        self.model.eval()  # Set model to evaluation mode
        sample_x = X[index].reshape(-1, 1, X.shape[-2])
        sample = UCRDataset(sample_x.astype(np.float32), y[index].astype(np.float32))
        input_tensor = torch.from_numpy(sample.feature).to(self.device)

        # Create storage for gradients
        gradients = {}

        # Register hook to capture gradients from conv2
        def save_gradients(module, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        # Hook the conv2 layer
        self.model.conv2.register_full_backward_hook(save_gradients)

        # Forward pass
        output = self.model(input_tensor)

        # Get the score for the target class
        score = output[:, target_class]

        # Backward pass to calculate gradients
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Extract gradients and feature maps
        activations = self.model.features.detach()  # Feature maps
        grads = gradients["value"].detach()  # Gradients w.r.t the feature maps

        # Compute weights (global average pooling of gradients)
        weights = torch.mean(grads, dim=2, keepdim=True)  # Average over time dimension

        # Compute Grad-CAM activation map
        grad_cam_map = torch.sum(weights * activations, dim=1).squeeze()  # Weighted sum of feature maps
        ### grad_cam_map = abs(grad_cam_map) ### 
        grad_cam_map = F.relu(grad_cam_map)  # Apply ReLU to keep only positive contributions

        # Upsample Grad-CAM map to match input size
        grad_cam_map = F.interpolate(
            grad_cam_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=(input_tensor.size(2)),  # Match the input time series length
            mode="linear",
            align_corners=False,
        ).squeeze()  # Remove extra dimensions

        # Normalize the Grad-CAM map
        grad_cam_map = grad_cam_map - grad_cam_map.min()
        grad_cam_map = grad_cam_map / grad_cam_map.max()
        return grad_cam_map.cpu().numpy()
    

    def save_model(self, filepath):
        # Save the model's state dictionary and other components
        torch.save({'model_state_dict': self.model.state_dict()}, filepath)


    def load_model(self, filepath):    
        # Load the model's state dictionary
        self.model.load_state_dict(torch.load(filepath)['model_state_dict'])
        
        # Move model to the appropriate device (CPU or GPU)
        self.model.to("cpu")  # Replace "cpu" with "cuda" if needed

        

        