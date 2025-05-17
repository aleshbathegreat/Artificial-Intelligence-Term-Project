import torch
import torch.nn as nn
import torch.optim as optim

#backpropagation-based training using PyTorch
def train_with_gradient_descent(model,X_train,y_train,
                                criterion=None,
                                learning_rate=0.001,
                                num_epochs=30,
                                batch_size=32):

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X_train,y_train)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X,batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs,batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}],Loss: {avg_loss:.4f}")

    return model
