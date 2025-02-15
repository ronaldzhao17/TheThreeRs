# import torch
# import torch.optim.lr_scheduler as lr_scheduler
# import torch.nn as nn
# import torch.optim as optim
# # import matplotlib.pyplot as plt
# from trash_model import TrashModel, num_classes
# from dataset import train_loader, test_loader


# def find_lr(model, train_loader, criterion, optimizer, start_lr=1e-6, end_lr=10, num_iters=100):
#     model.train()
#     lrs = torch.logspace(start=torch.log10(torch.tensor(start_lr)), 
#                          end=torch.log10(torch.tensor(end_lr)), 
#                          steps=num_iters).tolist()
    
#     losses = []
#     best_loss = float('inf')

#     for i, (inputs, labels) in enumerate(train_loader):
#         if i >= num_iters:
#             break
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.param_groups[0]['lr'] = lrs[i]
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         losses.append(loss.item())
#         if loss.item() < best_loss:
#             best_loss = loss.item()

#     return lrs, losses


# if __name__ == "__main__":
#     model = TrashModel(num_classes)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define Loss Function and Optimizer
#     criterion = nn.CrossEntropyLoss()  # Standard loss for classification tasks
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

#     # Run Learning Rate Finder
#     print("pre learning rate finder")
#     lrs, losses = find_lr(model, train_loader, criterion, optimizer)
#     print("done")

#     # Plot Learning Rate vs. Loss
#     # plt.figure(figsize=(8,6))
#     # plt.plot(lrs, losses)
#     # plt.xscale('log')
#     # plt.xlabel("Learning Rate")
#     # plt.ylabel("Loss")
#     # plt.title("Learning Rate Finder")
#     # plt.show()


#     # Define One-Cycle Learning Rate Scheduler
#     max_lr = 5e-3  # Equivalent to max_lr in FastAI
#     epochs = 20
#     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
#                                         steps_per_epoch=len(train_loader),
#                                         epochs=epochs)

#     # Function to compute accuracy
#     def compute_accuracy(model, data_loader):
#         model.eval()  # Set model to evaluation mode
#         correct = 0
#         total = 0
        
#         with torch.no_grad():  # No need to compute gradients for evaluation
#             for inputs, labels in data_loader:
#                 inputs, labels = inputs.to(model.device), labels.to(model.device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)  # Get class with highest probability
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         return 100 * correct / total  # Return accuracy percentage

#     # Training Loop with Test Accuracy
#     for epoch in range(epochs):
#         model.train()  # Set model to training mode
#         running_loss = 0.0
        
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(model.device), labels.to(model.device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             running_loss += loss.item()

#         # Compute average training loss
#         avg_loss = running_loss / len(train_loader)

#         # Compute test accuracy
#         test_accuracy = compute_accuracy(model, test_loader)

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

#     print("Training Completed! 🎉")
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.optim as optim
from trash_model import TrashModel, num_classes
from dataset import train_loader, test_loader


def find_lr(model, train_loader, criterion, optimizer, start_lr=1e-6, end_lr=10, num_iters=100):
    """ Implements Learning Rate Finder to identify optimal LR range. """
    model.train()
    lrs = torch.logspace(start=torch.log10(torch.tensor(start_lr)), 
                         end=torch.log10(torch.tensor(end_lr)), 
                         steps=num_iters).tolist()
    
    losses = []
    best_loss = float('inf')

    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_iters:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.param_groups[0]['lr'] = lrs[i]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

    return lrs, losses


if __name__ == "__main__":
    # ✅ Define device before initializing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Initialize model and move it to GPU/CPU
    model = TrashModel(num_classes).to(device)

    # ✅ Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Run Learning Rate Finder
    print("Running Learning Rate Finder...")
    lrs, losses = find_lr(model, train_loader, criterion, optimizer)
    print("LR Finder Completed!")

    # ✅ Reinitialize model & optimizer after LR Finder
    model = TrashModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Define One-Cycle Learning Rate Scheduler
    max_lr = 5e-3
    epochs = 20
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                        steps_per_epoch=len(train_loader),
                                        epochs=epochs)

    # ✅ Function to compute accuracy
    def compute_accuracy(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    # ✅ Training Loop with Test Accuracy
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_accuracy = compute_accuracy(model, test_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    print("Training Completed! 🎉")
