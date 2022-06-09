from utils import process_for_training, is_gan, is_noisegan, load_model, get_optimizer, get_criterion, process_for_eval
#from utils import train_shape_in, train_shape_out, val_shape_in, val_shape_out
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
    criterion_mr = get_criterion(args)
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
        running_loss_hr = 0
        running_loss_mr = 0
        #with tqdm(data[0], unit="batch") as tepoch: 
        k = 0 
        for (inputs,  targets) in data[0]:          
            inputs, targets = process_for_training(inputs, targets)
            if is_gan(args):
                loss, discr_loss = gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, data[0], args, criterion_mr)
                running_loss += loss
                running_discr_loss += discr_loss
            else:
                loss = optimizer_step(model, optimizer, criterion, inputs, targets, data[0], args, criterion_mr)

                running_loss += loss
                #print('train:',loss)
                #running_loss_mr += loss_mr
                #running_loss_hr += loss_hr
                #print('training', running_loss, loss, k)
            k +=1     
                
        #print('len', len(data))
        loss = running_loss/len(data[0])
        loss_mr = running_loss_mr/len(data[0])
        loss_hr = running_loss_hr/len(data[0])
        train_loss.append(loss)
        if is_gan(args):
            dicsr_loss = running_discr_loss/len(data)
            print('Epoch {}, Train Loss: {:.5f}, Discr. Loss{:.5f}'.format(
                epoch+1, loss, discr_loss))
            
            disc_loss.append(discr_loss)
        else:
            print('Epoch {}, Train Loss: {:.5f}'.format(epoch+1, loss))
            print(loss_mr, loss_hr)
            
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
    #if is_gan(args):
        #np.save(np.array(disc_loss), './data/losses/'+args.model_id+'-'+'disc_loss.npy')
        #np.save(np.array(train_loss_reg), './data/losses/'+args.model_id+'-'+'train_loss_reg.npy')
        #np.save(np.array(train_loss_adv), './data/losses/'+args.model_id+'-'+'train_loss_adv.npy')
        #np.save(np.array(val_loss_reg), './data/losses/'+args.model_id+'-'+'val_loss_reg.npy')
        #np.save(np.array(val_loss_adv), './data/losses/'+args.model_id+'-'+'val_loss_adv.npy')
    #else:
    np.save('./data/losses/'+args.model_id+'-'+'train_loss.npy', np.array(train_loss))
    np.save('./data/losses/'+args.model_id+'-'+'val_loss.npy', np.array(val_losses))
        
    

def optimizer_step(model, optimizer, criterion, inputs, targets, tepoch, args, criterion_mr=None, discriminator=False):
    optimizer.zero_grad()
    if args.mr:
        outputs, mr = model(inputs)
        mr_target = torch.nn.functional.avg_pool2d(targets[:,0,...], 2)
        loss_mr = criterion_mr(mr, mr_target.unsqueeze(1))
        loss_hr = criterion(outputs, targets)
        loss = args.alpha*loss_hr+ (1-args.alpha)*loss_mr
    else:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()  
    #print(torch.mean((outputs-targets)**2))
    #tepoch.set_postfix(loss=loss.item())
    return loss.item()#, loss_mr, loss_hr
    
    
def gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, tepoch, args):
    optimizer_discr.zero_grad()
    if args.noise:
        if False:
            z = np.random.normal( size=[inputs.shape[0], args.nsteps_in,args.nsteps_in,32,32])
            z_init = np.random.normal( size=[inputs.shape[0],args.nsteps_in,32,32])
            z = torch.Tensor(z).to(device)
            z_init = torch.Tensor(z_init).to(device)
            outputs = model(inputs, z, z_init)
        else:
            z = np.random.normal( size=[inputs.shape[0], 100])
            z = torch.Tensor(z).to(device)

            outputs = model(inputs, z)
    else:
        outputs = model(inputs)
    batch_size = inputs.shape[0]
    if False:
        real_label = torch.full((batch_size, args.nsteps_in, 1), 1, dtype=outputs.dtype).to(device)
        fake_label = torch.full((batch_size, args.nsteps_in, 1), 0, dtype=outputs.dtype).to(device)
    else:
        real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=outputs.dtype).to(device)

    # It makes the discriminator distinguish between real sample and fake sample.
    if False:
        real_output = discriminator_model(targets, inputs)
        fake_output = discriminator_model(outputs.detach(), inputs)
    else:
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
    if args.time:
        adversarial_loss = criterion_discr(discriminator_model(outputs, inputs), real_label)
    else:
        adversarial_loss = criterion_discr(discriminator_model(outputs), real_label)
    loss += args.adv_factor * adversarial_loss
    loss.backward()
    optimizer.step()       
    tepoch.set_postfix(loss=loss.item())
    return loss.item(), d_loss.item()

    
    
def validate_model(model, criterion, data, best, patience, epoch, args, discriminator_model=None, criterion_discr=None):
    model.eval()
    running_loss = 0    
    #with tqdm(data, unit="batch") as tepoch:       
    for i, (inputs, targets) in enumerate(data):     

        inputs, targets = process_for_training(inputs, targets)
        if is_gan(args):
            if args.noise:
                if False:
                    z = np.random.normal( size=[inputs.shape[0], 1,1,32,32])
                    z_init = np.random.normal( size=[inputs.shape[0],args.nsteps_in,32,32])
                    z = torch.Tensor(z).to(device)
                    z_init = torch.Tensor(z_init).to(device)
                    outputs = model(inputs, z, z_init)
                else:
                    z = np.random.normal( size=[inputs.shape[0], 100])
                    z = torch.Tensor(z).to(device)

                    outputs = model(inputs, z)
            else:
                if args.mr:
                    outputs, mr = model(inputs)
                else:
                    outputs = model(inputs)
            reg_loss = criterion(outputs, targets)
            loss = args.reg_factor*reg_loss
            batch_size = inputs.shape[0]
            if args.time:
                real_label = torch.full((batch_size, args.nsteps_out, 1), 1, dtype=outputs.dtype).to(device)
                fake_output = discriminator_model(outputs.detach(), inputs)
            else:
                real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
                fake_output = discriminator_model(outputs.detach())
            adversarial_loss = criterion_discr(fake_output.detach(), real_label)
            loss += args.adv_factor * adversarial_loss
        else:
            if args.mr:
                outputs, mr = model(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)  
        #print(torch.mean((outputs-targets)**2))
        running_loss += loss.item()
        #print('val:', loss.item())
        #print('val', running_loss, loss.item(), i)
    #print('len', len(data))
    loss = running_loss/len(data)
    model.train()
    return loss

Tensor = torch.cuda.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def checkpoint(model, val_loss, best, args):
    print(val_loss, best)
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
    running_mse = 0 
    running_ssim = 0
    running_mae = 0
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    ssim_criterion = tgm.losses.SSIM(window_size=11, max_val=data[4], reduction='mean')
    full_pred = torch.zeros(data[8]) ###!!!change back to data[8]  
    with tqdm(data[1], unit="batch") as tepoch:     ###!!!change back to data[1]  
            for i,(inputs,  targets) in enumerate(tepoch): 
                inputs, targets = process_for_training(inputs, targets)
                if args.noise:
                    if args.time:
                        z = np.random.normal( size=[inputs.shape[0], args.nsteps_in,args.nsteps_in,32,32])
                        z_init = np.random.normal( size=[inputs.shape[0],args.nsteps_in,32,32])
                        z = torch.Tensor(z).to(device)
                        z_init = torch.Tensor(z_init).to(device)
                        outputs = model(inputs, z, z_init)
                    else:
                        z = np.random.normal( size=[inputs.shape[0], 100])
                        z = torch.Tensor(z).to(device)

                        outputs = model(inputs, z)
                else:
                    if args.mr:
                        outputs, mr = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                outputs, targets = process_for_eval(outputs, targets,data[2], data[3], data[4], args) 
                print(full_pred.shape, outputs.shape, targets.shape)
                full_pred[i*args.batch_size:i*args.batch_size+outputs.shape[0],...] = outputs.detach().cpu()
                '''
                for j in range(targets.shape[1]):
                    
                    running_mse += l2_crit(outputs[:,j,...], targets[:,j,...]).item()
                    running_mae += l1_crit(outputs[:,j,...],targets[:,j,...]).item()                
                    running_ssim += ssim_criterion(outputs[:,j,...], targets[:,j,...]).item()       '''
                running_mse += l2_crit(outputs, targets).item()
                
                running_mae += l1_crit(outputs,targets).item()                
                #running_ssim += ssim_criterion(outputs, targets).item()
                                            
    '''                                       
    mse = running_mse/(len(data)*targets.shape[1])
    mae = running_mae/(len(data)*targets.shape[1])
    ssim = running_ssim/(len(data)*targets.shape[1])'''
    mse = running_mse/len(data)
    mae = running_mae/len(data)
    ssim = running_ssim/len(data)
    psnr = calculate_pnsr(mse, data[4])
    torch.save(full_pred, './data/prediction/'+args.dataset+'_'+args.model_id+'_fullprediction.pt')                                      
    return {'MSE':mse, 'RMSE':torch.sqrt(torch.Tensor([mse])), 'PSNR': psnr, 'MAE':mae, 'SSIM':1-ssim}


def evaluate_double_model(model1, model2, data, args):
    model1.eval()
    model2.eval()
    running_mse = 0    
    running_ssim = 0
    running_mae = 0
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    full_pred = torch.zeros(data[8])
    with tqdm(data[1], unit="batch") as tepoch:       
            for i,(inputs,  targets) in enumerate(tepoch): 
                inputs, targets = process_for_training(inputs, targets)
                if args.noise:
                    if args.time:
                        z = np.random.normal( size=[inputs.shape[0], args.nsteps_in,args.nsteps_in,32,32])
                        z_init = np.random.normal( size=[inputs.shape[0],args.nsteps_in,32,32])
                        z = torch.Tensor(z).to(device)
                        z_init = torch.Tensor(z_init).to(device)
                        outputs = model(inputs, z, z_init)
                    else:
                        z = np.random.normal( size=[inputs.shape[0], 100])
                        z = torch.Tensor(z).to(device)

                        outputs = model2(model1(inputs, z))
                else:
                    out = model1(inputs)
                    #x = torch.cat((inputs[:,0:1,...], out, inputs[:,1:2,...]), dim=1)
                    #if i ==0:
                    #    torch.save(x, './data/prediction/intermediate_'+args.dataset+'_'+args.model_id+'_'+args.model_id2+'.pt')
                    outputs = model2(out)
                outputs, targets = process_for_eval(outputs, targets,data[2], data[3], data[4], args) 
                full_pred[i*args.batch_size:i*args.batch_size+outputs.shape[0],...] = outputs.detach().cpu()
                running_mse += l2_crit(outputs, targets).item()
                
                running_mae += l1_crit(outputs,targets).item()                
                #running_ssim += ssim_criterion(outputs, targets).item()
                                            
    '''                                       
    mse = running_mse/(len(data)*targets.shape[1])
    mae = running_mae/(len(data)*targets.shape[1])
    ssim = running_ssim/(len(data)*targets.shape[1])'''
    mse = running_mse/len(data)
    mae = running_mae/len(data)
    ssim = running_ssim/len(data)
    psnr = 0
    psnr = calculate_pnsr(mse, data[4])
    torch.save(full_pred, './data/prediction/'+args.dataset+'_'+args.model_id+'_'+args.model_id2+'_fullprediction.pt')                                      
    return {'MSE':mse, 'RMSE':torch.sqrt(torch.Tensor([mse])), 'PSNR': psnr, 'MAE':mae, 'SSIM':1-ssim}
                                            

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




