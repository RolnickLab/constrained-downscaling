from utils import process_for_training, is_gan, is_noisegan, load_model, get_optimizer, get_criterion, process_for_eval, get_loss, load_data
import models
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchgeometry as tgm
import csv
import numpy as np
from scoring import main_scoring
device = 'cuda'

def run_training(args, data):
    model = load_model(args)
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)
    criterion = get_criterion(args)
    if is_gan(args):   
        discriminator_model = load_model(args, discriminator=True)
        print('#params discr.:', sum(p.numel() for p in discriminator_model.parameters()))
        optimizer_discr = get_optimizer(args, discriminator_model)
        criterion_discr = get_criterion(args, discriminator=True)      
    best = np.inf
    for epoch in range(args.epochs):
        running_loss = 0    
        running_discr_loss = 0
        running_adv_loss = 0
        running_mass_loss = 0
        for (inputs,  targets) in data[0]:          
            inputs, targets = process_for_training(inputs, targets)
            if is_gan(args):
                loss, discr_loss = gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, data[0], args, criterion_mr)
                running_loss += loss
                running_discr_loss += discr_loss
            else:
                loss = optimizer_step(model, optimizer, criterion, inputs, targets, data[0], args)
                running_loss += loss
            running_mass_loss += mass_loss
        loss = running_loss/len(data[0])
        if is_gan(args):
            dicsr_loss = running_discr_loss/len(data)
            print('Epoch {}, Train Loss: {:.5f}, Discr. Loss{:.5f}'.format(
                epoch+1, loss, discr_loss))      
            disc_loss.append(discr_loss)
        else:
            print('Epoch {}, Train Loss: {:.5f}'.format(epoch+1, loss))
            
        if is_gan(args):
            val_loss = validate_model(model, criterion, data[1], best, epoch, args, discriminator_model, criterion_discr)
        else:
            val_loss = validate_model(model, criterion, data[1], best, epoch, args)
        val_losses.append(val_loss)
        print('Val loss: {:.5f}'.format(val_loss))
        checkpoint(model, val_loss, best, args, epoch)
        best = np.minimum(best, val_loss)
    data = load_data(args)        
    scores = evaluate_model( data, args)
    
    
def optimizer_step(model, optimizer, criterion, inputs, targets, tepoch, args, discriminator=False):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = get_loss(outputs, targets, inputs,args)
    loss.backward()
    optimizer.step()  
    return loss.item()
    
def gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, tepoch, args):
    optimizer_discr.zero_grad()
    z = np.random.normal( size=[inputs.shape[0], 100])
    z = torch.Tensor(z).to(device)
    outputs = model(inputs, z)
    batch_size = inputs.shape[0]   
    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
    fake_label = torch.full((batch_size, 1), 0, dtype=outputs.dtype).to(device)   
    real_output = discriminator_model(targets)
    fake_output = discriminator_model(outputs.detach())
    # Adversarial loss for real and fake images  
    d_loss_real = criterion_discr(real_output, real_label)                    
    d_loss_fake = criterion_discr(fake_output, fake_label)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_discr.step() 
    optimizer.zero_grad()
    reg_loss = criterion(outputs, targets)
    loss = args.reg_factor*reg_loss
    # Adversarial loss for real and fake images (relativistic average GAN)
    adversarial_loss = criterion_discr(discriminator_model(outputs), real_label)
    loss += args.adv_factor * adversarial_loss
    loss.backward()
    optimizer.step()       
    return loss.item(), d_loss.item()
   
def validate_model(model, criterion, data, best, epoch, args, discriminator_model=None, criterion_discr=None):
    model.eval()
    running_loss = 0      
    for i, (inputs, targets) in enumerate(data):     
        inputs, targets = process_for_training(inputs, targets)
        if is_gan(args): 
            z = np.random.normal( size=[inputs.shape[0], 100])
            z = torch.Tensor(z).to(device)
            outputs = model(inputs, z)
            reg_loss = criterion(outputs, targets)
            loss = args.reg_factor*reg_loss
            batch_size = inputs.shape[0]
            real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
            fake_output = discriminator_model(outputs.detach())
            adversarial_loss = criterion_discr(fake_output.detach(), real_label)
            loss += args.adv_factor * adversarial_loss
        else:
             outputs = model(inputs)
            loss = get_loss(outputs, targets, inputs, args) 
        running_loss += loss.item()
    loss = running_loss/len(data)
    model.train()
    return loss

Tensor = torch.cuda.FloatTensor

def checkpoint(model, val_loss, best, args, epoch):
    print(val_loss, best)
    if val_loss < best:
        checkpoint = {'model': model,'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/'+args.model_id+'.pth')
        
def evaluate_model(data, args, add_string=None):
    model = load_model(args)
    load_weights(model, args.model_id)
    model.eval()
    full_pred = torch.zeros(data[8]) 
    with tqdm(data[1], unit="batch") as tepoch:     
        for i,(inputs,  targets) in enumerate(tepoch): 
            inputs, targets = process_for_training(inputs, targets)
            if is_gan(args):
                outputs = torch.zeros((targets.shape[0],10,1,1,targets.shape[3],targets.shape[4])).to(device)
                for j in range(10):
                    z = np.random.normal( size=[inputs.shape[0], 100])
                    z = torch.Tensor(z).to(device)
                    outputs[:,j,...] = model(inputs, z)
            else:
                if args.mr:
                    outputs, mr = model(inputs)
                else:
                    outputs = model(inputs)
            outputs, targets = process_for_eval(outputs, targets,data[2], data[3], data[4], args) 
            full_pred[i*args.batch_size:i*args.batch_size+outputs.shape[0],...] = outputs.detach().cpu()
    if is_gan(args):
        print('saving', full_pred.mean())
        torch.save(full_pred, './data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_ensemble.pt')
    else:
        torch.save(full_pred, './data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'.pt')
    calculate_scores(args)


def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                                            
def create_report(scores, args, add_string=None):
    args_dict = args_to_dict(args)
    #combine scorees and args dict
    args_scores_dict = args_dict | scores
    #save dict
    save_dict(args_scores_dict, args, add_string)
    
def args_to_dict(args):
    return vars(args)
    
                                            
def save_dict(dictionary, args, add_string):
    if add_string:
        w = csv.writer(open('./data/score_log/'+args.model_id+add_string+'.csv', 'w'))
    else:
        w = csv.writer(open('./data/score_log/'+args.model_id+'.csv', 'w'))
        
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])

def load_weights(model, model_id):
    PATH = '/home/harder/constraint_generative_ml/models/'+model_id+'.pth'
    checkpoint = torch.load(PATH) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    return model






