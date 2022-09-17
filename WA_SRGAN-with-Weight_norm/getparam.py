import torch

def print_first_parameter(net):	
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (str(name) + ':' + str(param.data[0]))
            return

        
# To check the parameters of the model in each epoch, the mean and max of the gradients should not be over 100. That means the progress in the training is leading to failure.
def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 100:
        print('WARNING!' + model_name + ' gradients mean is over 100.')
        return False
    if grads.any() and grads.max() > 100:
        print('WARNING!' + model_name + ' gradients max is over 100.')
        return False
    return True

# in each epoch the top and bottom parameters are dispalying to be sure the model is trainign properly
def get_grads_D(net,top_name,bottom_name):
    top = 0
    bottom = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            # Hardcoded param name, subject to change of the network
            if name == top_name:
                top = param.grad.abs().mean()

            # Hardcoded param name, subject to change of the network
            if name == bottom_name:
                bottom = param.grad.abs().mean()

    return top, bottom

# in each epoch the top and bottom parameters are dispalying to be sure the model is trainign properly
def get_grads_G(net,top_name,bottom_name):
    top =  torch.tensor([0])
    bottom =  torch.tensor([0])
    #torch.set_printoptions(precision=10)
    #torch.set_printoptions(threshold=50000)
    for name, param in net.named_parameters():
        if param.requires_grad:
            # Hardcoded param name, subject to change of the network
            if name == top_name:
                top = param.grad.abs().mean()

            # Hardcoded param name, subject to change of the network
            if name == bottom_name:
                bottom = param.grad.abs().mean()

    return top, bottom

# return the trainable parameters 
def get_param_names(net):
    param_str=[]
    for name,param in net.named_parameters():
        if param.requires_grad:
            if 'weight' in name:
                param_str.append(name)
    return param_str[0],param_str[-1]

