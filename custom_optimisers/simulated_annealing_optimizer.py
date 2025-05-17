import torch
import numpy as np

#functions for FNN
def get_flat_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def set_flat_params(model,flat_params):
    pointer =0
    for param in model.parameters():
        num_params = param.numel()
        param.data.copy_(flat_params[pointer:pointer+num_params].view_as(param))
        pointer += num_params

def evaluate(model,X,y,criterion):
    model.eval()
    with torch.no_grad():
        outputs =model(X)
        loss = criterion(outputs,y)
    return loss.item()

#simulated annealing main function
def simulated_annealing(model,X_train,y_train,criterion,Temperature_init=1.0,Temperature_min=1e-3,alpha=0.95,max_iter=1000):
    current_params = get_flat_params(model).clone()
    best_params = current_params.clone()
    current_loss = evaluate(model,X_train,y_train,criterion)
    best_loss = current_loss
    T = Temperature_init

    for i in range(max_iter):
        new_params = current_params +torch.randn_like(current_params)*0.1
        set_flat_params(model,new_params)

        new_loss = evaluate(model,X_train,y_train,criterion)
        delta = new_loss-current_loss

        if delta<0 or np.random.rand()<np.exp(-delta / T):
            current_params = new_params
            current_loss = new_loss
            if new_loss< best_loss:
                best_loss = new_loss
                best_params = new_params.clone()
        else:
            set_flat_params(model,current_params)

        T *= alpha
        if T < Temperature_min:
            break

        if i% 100 == 0:
            print(f"Iter {i}, Temp: {T:.4f}, Loss: {current_loss:.4f}")

    set_flat_params(model,best_params)
    return model
