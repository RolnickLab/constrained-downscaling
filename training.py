from utils import process_for_training, is_gan, is_noisegan, load_model, get_optimizer, get_criterion, process_for_eval
import models
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchgeometry as tgm
import csv
import numpy as np
device = 'cuda'
#torch.set_default_dtype(torch.float64)
def run_training(args, data):
    model = load_model(args)
    optimizer = get_optimizer(args, model)
    criterion = get_criterion(args)
    if is_gan(args):
        
        discriminator_model = load_model(args, discriminator=True)
        optimizer_discr = get_optimizer(args, discriminator_model)
        criterion_discr = get_criterion(args, discriminator=True)
    best = np.inf
    patience_count = 0 
    is_stop = False
    val_losses = []
    train_loss = []
    if is_gan(args):
        disc_loss = []
        train_loss_reg = []
        train_loss_adv = []
        val_loss_reg = []
        val_loss_adv = []
    for epoch in range(args.epochs):
        running_loss = 0    
        running_discr_loss = 0
        running_adv_loss = 0
        with tqdm(data[0], unit="batch") as tepoch:       
            for (inputs, targets) in tepoch:          
                inputs, targets = process_for_training(inputs, targets)
                if is_gan(args):
                    loss, discr_loss = gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, tepoch, args)
                    running_loss += loss
                    running_discr_loss += discr_loss
                else:
                    loss = optimizer_step(model, optimizer, criterion, inputs, targets, tepoch, args)
                    running_loss += loss
        loss = running_loss/len(data)
        train_loss.append(loss)
        if is_gan(args):
            dicsr_loss = running_discr_loss/len(data)
            print('Epoch {}, Train Loss: {:.5f}, Discr. Loss{:.5f}'.format(
                epoch+1, loss, discr_loss))
            disc_loss.append(discr_loss)
        else:
            print('Epoch {}, Train Loss: {:.5f}'.format(epoch+1, loss))
            
        if is_gan(args):
            val_loss = validate_model(model, criterion, data[1], best, patience_count, epoch, args, discriminator_model, criterion_discr)
        else:
            val_loss = validate_model(model, criterion, data[1], best, patience_count, epoch, args)
        val_losses.append(val_loss)
        print('Val loss: {:.5f}'.format(val_loss))
        checkpoint(model, val_loss, best, args)
        #if args.early_stop:
         #   is_stop, patience_count = check_for_early_stopping(val_loss, best, patience_count, args)
        best = np.minimum(best, val_loss)
        if is_stop:
            break
    scores = evaluate_model(model, data, args)
    print(scores)
    create_report(scores, args)
    if is_gan(args):
        np.save(np.array(disc_loss), './data/losses/'+args.model_id+'-'+'disc_loss.npy')
        np.save(np.array(train_loss_reg), './data/losses/'+args.model_id+'-'+'train_loss_reg.npy')
        np.save(np.array(train_loss_adv), './data/losses/'+args.model_id+'-'+'train_loss_adv.npy')
        np.save(np.array(val_loss_reg), './data/losses/'+args.model_id+'-'+'val_loss_reg.npy')
        np.save(np.array(val_loss_adv), './data/losses/'+args.model_id+'-'+'val_loss_adv.npy')
    else:
        np.save(np.array(train_loss), './data/losses/'+args.model_id+'-'+'train_loss.npy')
        np.save(np.array(val_losses), './data/losses/'+args.model_id+'-'+'val_loss.npy')
        
    

def optimizer_step(model, optimizer, criterion, inputs, targets, tepoch, args, discriminator=False):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()       
    tepoch.set_postfix(loss=loss.item())
    return loss.item()
    
    
def gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, tepoch, args):
    optimizer_discr.zero_grad()
    if args.noise:
        z = np.random.normal( size=[inputs.shape[0], 100])
        z = torch.Tensor(z).to(device)

        outputs = model(inputs, z)
    else:
        outputs = model(inputs)
    batch_size = inputs.shape[0]
    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
    fake_label = torch.full((batch_size, 1), 0, dtype=outputs.dtype).to(device)

    # It makes the discriminator distinguish between real sample and fake sample.
    real_output = discriminator_model(targets)
    fake_output = discriminator_model(outputs.detach())
    # Adversarial loss for real and fake images                   
    d_loss_real = criterion_discr(real_output, real_label)                    
    d_loss_fake = criterion_discr(fake_output, fake_label)
    # Count all discriminator losses.
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_discr.step()
    
    optimizer.zero_grad()
    #outputs = model(inputs) ?
    reg_loss = criterion(outputs, targets)
    loss = args.reg_factor*reg_loss
    # Adversarial loss for real and fake images (relativistic average GAN)
    adversarial_loss = criterion_discr(discriminator_model(outputs), real_label)
    loss += args.adv_factor * adversarial_loss
    loss.backward()
    optimizer.step()       
    tepoch.set_postfix(loss=loss.item())
    return loss.item(), d_loss.item()

    
    
def validate_model(model, criterion, data, best, patience, epoch, args, discriminator_model=None, criterion_discr=None):
    model.eval()
    running_loss = 0    
    with tqdm(data, unit="batch") as tepoch:       
        for i, (inputs, targets) in enumerate(tepoch):          
            inputs, targets = process_for_training(inputs, targets)
            if is_gan(args):
                if args.noise:
                    z = np.random.normal( size=[inputs.shape[0], 100])
                    z = torch.Tensor(z).to(device)

                    outputs = model(inputs, z)
                else:
                    outputs = model(inputs)
                reg_loss = criterion(outputs, targets)
                loss = args.reg_factor*reg_loss
                batch_size = inputs.shape[0]
                real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
                fake_output = discriminator_model(outputs.detach()) 
                adversarial_loss = criterion_discr(fake_output.detach(), real_label)
                loss += args.adv_factor * adversarial_loss
            else:
                outputs = model(inputs)
                if i == 0:
                    torch.save(outputs[0,0,:,:],'./data/images/'+args.model_id+'_'+str(epoch)+'.pt')
                loss = criterion(outputs, targets)            
            running_loss += loss.item()
    loss = running_loss/len(data)
    model.train()
    return loss


def checkpoint(model, val_loss, best, args):
    if val_loss < best:
        checkpoint = {'model': model,'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/'+args.model_id+'.pth')
        
        
def check_for_early_stopping(val_loss, best, patience_counter, args):
    is_stop = False
    if val_loss < best:
        patience_counter = 0
    else:
        patience_counter+=1
    if patience_counter == args.patience:
        is_stop = True 
    return is_stop, patience_counter



def evaluate_model(model, data, args):
    model.eval()
    running_mse = np.zeros((args.dim,1))    
    running_ssim = np.zeros((args.dim,1))
    running_mae = np.zeros((args.dim,1)) 
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    
    with tqdm(data[1], unit="batch") as tepoch:       
            for i,(inputs, targets) in enumerate(tepoch): 
                inputs, targets = process_for_training(inputs, targets)
                if args.noise:
                    z = np.random.normal( size=[inputs.shape[0], 100])
                    z = torch.Tensor(z).to(device)

                    outputs = model(inputs, z)
                else:
                    outputs = model(inputs)
                outputs, targets = process_for_eval(outputs, targets,data[2], data[3], data[4], args) 
                print(outputs.shape, targets.shape)
                if i == 0:
                    torch.save(outputs, './data/prediction/'+args.dataset+'_'+args.model_id+'_prediction.pt')
                for j in range(args.dim):
                    ssim_criterion = tgm.losses.SSIM(window_size=11, max_val=data[4][j], reduction='mean')
                    running_mse[j] += l2_crit(outputs[:,j,:,:], targets[:,j,:,:]).item()
                    running_mae[j] += l1_crit(outputs[:,j,:,:],targets[:,j,:,:]).item()

                    running_ssim[j] += ssim_criterion(outputs[:,j,:,:].unsqueeze(1), targets[:,j,:,:].unsqueeze(1)).item()
                                            
                                            
    mse = running_mse/len(data)
    mae = running_mae/len(data)
    ssim = running_ssim/len(data)
    psnr = np.zeros((args.dim,1))
    for i in range(args.dim):
        psnr[i] = calculate_pnsr(mse[i], data[4][i])
                                            
    return {'MSE':mse, 'RMSE':torch.sqrt(torch.Tensor([mse])), 'PSNR': psnr, 'MAE':mae, 'SSIM':np.ones((args.dim,1))-ssim}
                                            

def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                                            
def create_report(scores, args):
    args_dict = args_to_dict(args)
    #combine scorees and args dict
    args_scores_dict = args_dict | scores
    #save dict
    save_dict(args_scores_dict, args)
    
def args_to_dict(args):
    return vars(args)
    
                                            
def save_dict(dictionary, args):
    w = csv.writer(open('./data/score_log/'+args.model_id+'.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])




