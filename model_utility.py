import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch import optim
from collections import OrderedDict

def Build_Train_Model(data_dir,arch="vgg19",learning_rate=0.003,hidden_units=1024,epochs=30,use_gpu = False ,save_dir="NULL"):
    
    #get data
    data_collection = Get_Data(data_dir)
    dataloaders = data_collection['dataloaders']

    #create model
    model = Get_Model(arch)
    model.arch = arch

    #device
    device = torch.device('cpu')
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device: {0}".format(device.type))
    #swicth off gradients for now
    for param in model.parameters():
        param.requires_grad = False
    
    #we need to know the number of classes or outputs
    model.class_to_idx = data_collection['datasets']['train'].class_to_idx
    class_count = 0
    for a,e in model.class_to_idx.items():
        class_count+=1

    #number of output units
    output_units = class_count
    model.output_units = output_units

    #number of input units
    input_units = Get_Model_Inputs(arch)
    model.input_units = input_units

    #classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('fc2', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_units, output_units)),
                          ('fc6', nn.LogSoftmax(dim=1)),
                          ]))
    model.classifier = classifier

    #loss
    criterion = nn.NLLLoss()

    # optimizer
    optimizer = optim.SGD(model.classifier.parameters(),lr = learning_rate)

    model.to(device)

    #TRAIIN THE MODEL
    #train for 30 time periods
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            
            #reset the initial grad to prevent grad accumulation from previous batch
            optimizer.zero_grad()
            
            #send inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            #forward pass
            logps = model.forward(inputs)
            
            #loss
            loss = criterion(logps,labels)
            
            #backward pass
            loss.backward()
            
            #optimization step
            optimizer.step()
            
            running_loss += loss.item()
        else:
            validation_loss = 0
            accuracy = 0
            
            #switch from training mode to evaluation mode
            model.eval()
            
            #use a scope where grad is off
            with torch.no_grad():
                for inputs, labels in dataloaders['validation']:
                    #send inputs and labels to device
                    inputs,labels = inputs.to(device), labels.to(device)
                    
                    #forward pass
                    logps = model.forward(inputs)
                    
                    #loss
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()
                    
                    #probabilities from log prob.
                    ps = torch.exp(logps)
                    
                    #Get the top (1) prediction and class
                    top_p, top_class = ps.topk(1,dim=1)
                    
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
                    f"Validation loss: {validation_loss/len(dataloaders['validation']):.3f}.. "
                    f"Test accuracy: {accuracy/len(dataloaders['validation']):.3f}")
                
                running_loss = 0
                
                #switch back to training mode
                model.train()
    
    if save_dir != 'NULL':
        print('saving checkpoint')
        SaveCheckpoint(model, save_dir)

def SaveCheckpoint(model, save_dir):
    classifier_attr = list()

    #save the details of the classifier
    for each in model.classifier:
        classifier_attr.append(str(each))

    checkpoint = {'input_size': model.input_units,
                'arch': model.arch,
              'output_size': model.output_units,
              'classifier': classifier_attr,
              'class-to-index': model.class_to_idx,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def LoadCheckpoint(checkpoint_path):
    #load checkpoint
    mycheckpoint = torch.load(checkpoint_path)
    
    #create the model again
    model = Get_Model(mycheckpoint['arch'])
    model.arch = mycheckpoint['arch']
    #number of output units
    model.output_units = int(mycheckpoint['output_size'])

    #number of input units
    model.input_units = int(mycheckpoint['input_size'])
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    myorderedDict = OrderedDict()
    
    fcname = 'fc'
    index = 0
    
    #get ordered dict of transformations for classifier
    for input in mycheckpoint['classifier']:
        index+=1
        if input.startswith('Linear'):
            myorderedDict[fcname+str(index)] = GetLinear(input)
        elif input.startswith('ReLU') or input.startswith('Sig'):
            myorderedDict[fcname+str(index)] = GetActivation(input)
        elif input.startswith('LogSoftmax') or input.startswith('Softmax'):
            myorderedDict[fcname+str(index)] = GetSoftmax(input)
    
        
    #create classifier
    classifier = nn.Sequential(myorderedDict)
    model.classifier = classifier
    
    #model state
    model.load_state_dict(mycheckpoint['state_dict'])
    
    #class to index mapping that was used when the models where trained
    model.class_to_idx = mycheckpoint['class-to-index']
    
    return model

def GetSoftmax(input):
    if input.startswith('LogSoftmax'):
        return nn.LogSoftmax(dim=1)
    else:
        return nn.Softmax(dim=1)

def GetActivation(input):
    if input.startswith('ReLU'):
        return nn.ReLU()
    elif input.startswith('Sig'):
        return nn.Sigmoid()

def GetLinear(input):
    in_features = 0
    out_features = 0
    bias = True
    if input.startswith('Linear'):
        argstr = input.split('(')
        prmstr = argstr[1][0:len(argstr[1])-1:1]
        paramlist = prmstr.split(',')
        for prm in paramlist:
            prm = prm.strip()
            if prm.startswith('in_features'):
                in_features = int(prm.split('=')[1])
            elif prm.startswith('out_features'):
                out_features = int(prm.split('=')[1])
            else:
                bias = bool(prm.split('=')[1])
                
    return nn.Linear(in_features, out_features, bias)

def Get_Data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train':transforms.Compose([
        transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'validation':transforms.Compose([
        transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose([
        transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train':datasets.ImageFolder(train_dir, transform= data_transforms['train']),
        'validation':datasets.ImageFolder(valid_dir, transform= data_transforms['validation']),
        'test':datasets.ImageFolder(test_dir, transform= data_transforms['test'])
    }


    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32,shuffle=True),
        'validation':torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 32),
        'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
    }

    data_collection = {
        'transforms':data_transforms,
        'datasets':image_datasets,
        'dataloaders':dataloaders
    }

    return data_collection



def Get_Model(arch="vgg19"):
    arch = arch.strip()
    if arch.lower() == 'vgg19':
        return models.vgg19(pretrained = True)
    elif arch.lower() == 'alexnet':
        return models.alexnet(pretrained = True)
    elif arch.lower() == 'vgg11':
        return models.vgg11(pretrained = True)
    elif arch.lower() == 'vgg11_bn':
        return models.vgg11_bn(pretrained = True)
    elif arch.lower() == 'vgg13':
        return models.vgg13(pretrained = True)
    elif arch.lower() == 'vgg13_bn':
        return models.vgg13_bn(pretrained = True)
    elif arch.lower() == 'vgg16':
        return models.vgg16(pretrained = True)
    elif arch.lower() == 'vgg16_bn':
        return models.vgg16_bn(pretrained = True)
    elif arch.lower() == 'vgg19_bn':
        return models.vgg19_bn(pretrained = True)
    elif arch.lower() == 'resnet18':
        return models.resnet18(pretrained = True)
    elif arch.lower() == 'resnet34':
        return models.resnet34(pretrained = True)
    elif arch.lower() == 'resnet50':
        return models.resnet50(pretrained = True)
    elif arch.lower() == 'resnet10':
        return models.resnet10(pretrained = True)
    elif arch.lower() == 'resnet152':
        return models.resnet152(pretrained = True)
    elif arch.lower() == 'squeezenet1_0':
        return models.squeezenet1_0(pretrained = True)
    elif arch.lower() == 'squeezenet1_1':
        return models.squeezenet1_1(pretrained = True)
    elif arch.lower() == 'densenet121':
        return models.densenet121(pretrained = True)
    elif arch.lower() == 'densenet169':
        return models.densenet169(pretrained = True)
    elif arch.lower() == 'densenet161':
        return models.densenet161(pretrained = True)
    elif arch.lower() == 'densenet201':
        return models.densenet201(pretrained = True)
    elif arch.lower() == 'inception_v3':
        return models.inception_v3(pretrained = True)
    elif arch.lower() == 'googlenet':
        return models.googlenet(pretrained = True)
    elif arch.lower() == 'shufflenet_v2_x0_5':
        return models.shufflenet_v2_x0_5(pretrained = True)
    elif arch.lower() == 'shufflenet_v2_x1_0':
        return models.shufflenet_v2_x1_0(pretrained = True)
    elif arch.lower() == 'shufflenet_v2_x1_5':
        return models.shufflenet_v2_x1_5(pretrained = True)
    elif arch.lower() == 'shufflenet_v2_x2_0':
        return models.shufflenet_v2_x2_0(pretrained = True)
    elif arch.lower() == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained = True)
    elif arch.lower() == 'mobilenet_v3_large':
        return models.mobilenet_v3_large(pretrained = True)
    elif arch.lower() == 'mobilenet_v3_small':
        return models.mobilenet_v3_small(pretrained = True)
    elif arch.lower() == 'resnext50_32x4d':
        return models.resnext50_32x4d(pretrained = True)
    elif arch.lower() == 'resnext101_32x8d':
        return models.resnext101_32x8d(pretrained = True)
    elif arch.lower() == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained = True)
    elif arch.lower() == 'wide_resnet101_2':
        return models.wide_resnet101_2(pretrained = True)
    elif arch.lower() == 'mnasnet0_5':
        return models.mnasnet0_5(pretrained = True)
    elif arch.lower() == 'mnasnet0_75':
        return models.mnasnet0_75(pretrained = True)
    elif arch.lower() == 'mnasnet1_0':
        return models.mnasnet1_0(pretrained = True)
    elif arch.lower() == 'mnasnet1_3':
        return models.mnasnet1_3(pretrained = True)
    elif arch.lower() == 'efficientnet_b0':
        return models.efficientnet_b0(pretrained = True)
    elif arch.lower() == 'efficientnet_b1':
        return models.efficientnet_b1(pretrained = True)
    elif arch.lower() == 'efficientnet_b2':
        return models.efficientnet_b2(pretrained = True)
    elif arch.lower() == 'efficientnet_b3':
        return models.efficientnet_b3(pretrained = True)
    elif arch.lower() == 'efficientnet_b4':
        return models.efficientnet_b4(pretrained = True)
    elif arch.lower() == 'efficientnet_b5':
        return models.efficientnet_b5(pretrained = True)
    elif arch.lower() == 'efficientnet_b6':
        return models.efficientnet_b6(pretrained = True)
    elif arch.lower() == 'efficientnet_b7':
        return models.efficientnet_b7(pretrained = True)
    elif arch.lower() == 'efficientnet_v2_s':
        return models.efficientnet_v2_s(pretrained = True)
    elif arch.lower() == 'efficientnet_v2_m':
        return models.efficientnet_v2_m(pretrained = True)
    elif arch.lower() == 'efficientnet_v2_l':
        return models.efficientnet_v2_l(pretrained = True)
    elif arch.lower() == 'regnet_y_400mf':
        return models.regnet_y_400mf(pretrained = True)
    elif arch.lower() == 'regnet_y_800mf':
        return models.regnet_y_800mf(pretrained = True)
    elif arch.lower() == 'regnet_y_1_6gf':
        return models.regnet_y_1_6gf(pretrained = True)
    elif arch.lower() == 'regnet_y_3_2gf':
        return models.regnet_y_3_2gf(pretrained = True)
    elif arch.lower() == 'regnet_y_8gf':
        return models.regnet_y_8gf(pretrained = True)
    elif arch.lower() == 'regnet_y_16gf':
        return models.regnet_y_16gf(pretrained = True)
    elif arch.lower() == 'regnet_y_32gf':
        return models.regnet_y_32gf(pretrained = True)
    elif arch.lower() == 'regnet_y_128gf':
        return models.regnet_y_128gf(pretrained = True)
    elif arch.lower() == 'regnet_x_400mf':
        return models.regnet_x_400mf(pretrained = True)
    elif arch.lower() == 'regnet_x_800mf':
        return models.regnet_x_800mf(pretrained = True)
    elif arch.lower() == 'regnet_x_1_6gf':
        return models.regnet_x_1_6gf(pretrained = True)
    elif arch.lower() == 'regnet_x_3_2gf':
        return models.regnet_x_3_2gf(pretrained = True)
    elif arch.lower() == 'regnet_x_8gf':
        return models.regnet_x_8gf(pretrained = True)
    elif arch.lower() == 'regnet_x_16gf':
        return models.regnet_x_16gf(pretrained = True)
    elif arch.lower() == 'regnet_x_32gf':
        return models.regnet_x_32gf(pretrained = True)
    elif arch.lower() == 'vit_b_16':
        return models.vit_b_16(pretrained = True)
    elif arch.lower() == 'vit_b_32':
        return models.vit_b_32(pretrained = True)
    elif arch.lower() == 'vit_l_16':
        return models.vit_l_16(pretrained = True)
    elif arch.lower() == 'vit_l_32':
        return models.vit_l_32(pretrained = True)
    elif arch.lower() == 'vit_h_14':
        return models.vit_h_14(pretrained = True)
    elif arch.lower() == 'convnext_tiny':
        return models.convnext_tiny(pretrained = True)
    elif arch.lower() == 'convnext_small':
        return models.convnext_small(pretrained = True)
    elif arch.lower() == 'convnext_base':
        return models.convnext_base(pretrained = True)
    elif arch.lower() == 'convnext_large':
        return models.convnext_large(pretrained = True)
    elif arch.lower() == 'swin_t':
        return models.swin_t(pretrained = True)

def Get_Model_Inputs(arch="vgg19"):
    arch = arch.strip()
    if arch.lower() == 'vgg19':
        return 25088
    elif arch.lower() == 'alexnet':
        return 9216
    elif arch.lower() == 'vgg11':
        return 25088
    elif arch.lower() == 'vgg11_bn':
        return 25088
    elif arch.lower() == 'vgg13':
        return 25088
    elif arch.lower() == 'vgg13_bn':
        return 25088
    elif arch.lower() == 'vgg16':
        return 25088
    elif arch.lower() == 'vgg16_bn':
        return 25088
    elif arch.lower() == 'vgg19_bn':
        return 25088
    elif arch.lower() == 'resnet18':
        return 512
    elif arch.lower() == 'resnet34':
        return 512
    elif arch.lower() == 'resnet50':
        return 2048
    elif arch.lower() == 'resnet10':
        return 512
    elif arch.lower() == 'resnet152':
        return 2048
    elif arch.lower() == 'squeezenet1_0':
        return 512
    elif arch.lower() == 'squeezenet1_1':
        return 512
    elif arch.lower() == 'densenet121':
        return 1024
    elif arch.lower() == 'densenet169':
        return 1664
    elif arch.lower() == 'densenet161':
        return 2208
    elif arch.lower() == 'densenet201':
        return 1920
    elif arch.lower() == 'inception_v3':
        return 2048
    elif arch.lower() == 'googlenet':
        return 1024
    elif arch.lower() == 'shufflenet_v2_x0_5':
        return 1024
    elif arch.lower() == 'shufflenet_v2_x1_0':
        return 1024
    elif arch.lower() == 'shufflenet_v2_x1_5':
        return 1024
    elif arch.lower() == 'shufflenet_v2_x2_0':
        return 1024
    elif arch.lower() == 'mobilenet_v2':
        return 1280
    elif arch.lower() == 'mobilenet_v3_large':
        return 960
    elif arch.lower() == 'mobilenet_v3_small':
        return 576
    elif arch.lower() == 'resnext50_32x4d':
        return 2048
    elif arch.lower() == 'resnext101_32x8d':
        return 2048
    elif arch.lower() == 'wide_resnet50_2':
        return 2048
    elif arch.lower() == 'wide_resnet101_2':
        return 2048
    elif arch.lower() == 'mnasnet0_5':
        return 1280
    elif arch.lower() == 'mnasnet0_75':
        return 1280
    elif arch.lower() == 'mnasnet1_0':
        return 1280
    elif arch.lower() == 'mnasnet1_3':
        return 1280
    elif arch.lower() == 'efficientnet_b0':
        return 1280
    elif arch.lower() == 'efficientnet_b1':
        return 1280
    elif arch.lower() == 'efficientnet_b2':
        return 1408
    elif arch.lower() == 'efficientnet_b3':
        return 1536
    elif arch.lower() == 'efficientnet_b4':
        return 1792
    elif arch.lower() == 'efficientnet_b5':
        return 2048
    elif arch.lower() == 'efficientnet_b6':
        return 2304
    elif arch.lower() == 'efficientnet_b7':
        return 2560
    elif arch.lower() == 'efficientnet_v2_s':
        return 1280
    elif arch.lower() == 'efficientnet_v2_m':
        return 1280
    elif arch.lower() == 'efficientnet_v2_l':
        return 1280
    elif arch.lower() == 'regnet_y_400mf':
        return 440
    elif arch.lower() == 'regnet_y_800mf':
        return 784
    elif arch.lower() == 'regnet_y_1_6gf':
        return 888
    elif arch.lower() == 'regnet_y_3_2gf':
        return 1512
    elif arch.lower() == 'regnet_y_8gf':
        return 2016
    elif arch.lower() == 'regnet_y_16gf':
        return 3024
    elif arch.lower() == 'regnet_y_32gf':
        return 3712
    elif arch.lower() == 'regnet_y_128gf':
        return 3712
    elif arch.lower() == 'regnet_x_400mf':
        return 400
    elif arch.lower() == 'regnet_x_800mf':
        return 672
    elif arch.lower() == 'regnet_x_1_6gf':
        return 912
    elif arch.lower() == 'regnet_x_3_2gf':
        return 1008
    elif arch.lower() == 'regnet_x_8gf':
        return 1920
    elif arch.lower() == 'regnet_x_16gf':
        return 2048
    elif arch.lower() == 'regnet_x_32gf':
        return 2520
    elif arch.lower() == 'vit_b_16':
        return 768
    elif arch.lower() == 'vit_b_32':
        return 768
    elif arch.lower() == 'vit_l_16':
        return 1024
    elif arch.lower() == 'vit_l_32':
        return 1024
    elif arch.lower() == 'vit_h_14':
        return 1024
    elif arch.lower() == 'convnext_tiny':
        return 768
    elif arch.lower() == 'convnext_small':
        return 768
    elif arch.lower() == 'convnext_base':
        return 1024
    elif arch.lower() == 'convnext_large':
        return 1536
    elif arch.lower() == 'swin_t':
        return 1024

