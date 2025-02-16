import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing

# Important: Guard your main code so that on Windows, the spawn method doesn’t re-run everything.
if __name__ == "__main__":
    # Set the start method for multiprocessing (Windows requires this)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Import after setting the start method
    from trash_model import TrashModel
    from dataset import classes, train_loader, test_loader, num_classes

    # Define device before initializing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with the correct number of classes and move it to the selected device
    model = TrashModel(num_classes).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Learning Rate Finder Function
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

    # print("Running Learning Rate Finder...")
    # lrs, losses = find_lr(model, train_loader, criterion, optimizer)
    # print("LR Finder Completed!")

    # Reinitialize model & optimizer after LR Finder so that LR Finder does not affect training.
    model = TrashModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define One-Cycle Learning Rate Scheduler
    max_lr = 5e-3
    epochs = 35
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                        steps_per_epoch=len(train_loader),
                                        epochs=epochs)

    # Function to compute accuracy
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

    torch.save(model.state_dict(), "trash_model.pth")

    # -----------------------
    # Compute and Display Confusion Matrix
    # -----------------------
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues,
                xticklabels=[classes[i] for i in range(num_classes)],
                yticklabels=[classes[i] for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

