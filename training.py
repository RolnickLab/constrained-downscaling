from utils import process_for_training, is_gan, is_noisegan, load_model, get_optimizer, get_criterion
import models

def run_training(args, data):
    model = load_model(args)
    optmizer = get_optimizer(args)
    criterion = get_criterion(args)
    if is_gan(args):
        discriminator_model = load_model(args, discriminator=True)
        discriminator_criterion = get_criterion(args, discriminator=True)
    best = np.inf
    patience_count = 0 
    for epoch in range(args.epochs):
        running_loss = 0    
        running_discr_loss = 0
        with tqdm(data[0], unit="batch") as tepoch:       
            for (inputs, targets) in tepoch:          
                inputs, targets = process_for_training(inputs, targets, args)
                if is_gan(args):
                    loss, discr_loss = gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, args)
                    running_loss += loss
                    running_discr_loss += discr_loss
                else:
                    loss = optimizer_step(model, optimizer, criterion, inputs, targets, tepoch, args)
                    running_loss += loss
        loss = running_loss/len(data)
        if is_gan(args):
            dicsr_loss = running_discr_loss/len(data)
        print('Epoch {}, Train Loss: {:.5f}, Discr. Loss{:.5f}'.format(
            epoch+1, loss, discrr_loss))

        val_loss = validate_model(model, criterion, data[1], best, patience)
        print('Val loss: {:.5f}'.format(val_loss))
        checkpoint(model, val_loss, args)
        if args.early_stop:
            is_stop, patience = check_for_early_stopping(val_loss, best, patience)
        best = np.minimum(best, val_loss)
        if is_stop:
            break

def optimizer_step(model, optmizer, criterion, inputs, targets, tepoch, args, discriminator=False):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()       
    tepoch.set_postfix(loss=loss.item())
    return loss.item()
    
def gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, criterion, criterion_discr, inputs, targets, args):
    optimizer_discr.zero_grad()
    outputs = model(inputs)
    batch_size = inputs.shape[0]
    real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).to(device)
    fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).to(device)

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
    outputs = model(inputs)
    reg_loss = criterion(outputs, targets)
    loss = args.reg_factor*reg_loss
    # Adversarial loss for real and fake images (relativistic average GAN)
    adversarial_loss = criterion_discr(discriminator_model(outputs), real_label)
    loss += args.adv_factor * adversarial_loss
    loss.backward()
    optimizer.step()       
    tepoch.set_postfix(loss=loss.item())
    return loss.item(), d_loss.item()
    
    
def validate_model(model, criterion, data, best, patience):
    model.eval()
    running_loss = 0    
    with tqdm(data, unit="batch") as tepoch:       
        for (inputs, targets) in tepoch:          
            inputs, targets = process_for_training(inputs, targets, args)
            if is_gan(args):
                outputs = model(inputs)
                reg_loss = criterion(outputs, targets)
                loss = args.reg_factor*reg_loss
                batch_size = inputs.shape[0]
                real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).to(device)
                fake_output = discriminator_model(outputs.detach()) 
                adversarial_loss = adversarial_criterion(fake_output.detach(), real_label)
                loss += args.adv_factor * adversarial_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)            
            running_loss += loss.item()
    loss = running_loss/len(data)
    model.train()
    return loss

def checkpoint(model, val_loss, best, args):
    if val_loss < best:
        checkpoint = {'model': model,'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/'+args.model_id+'.pth')
        
def check_for_early_stopping(val_loss, best, patience_counter):
    is_stop = False
    if val_loss < best:
        patience_counter = 0
    else:
        patience_counter+=1
    if patience_counter == args.patience:
        is_stop = True 
    return is_stop, patience
    
    
def save_args():
    pass
    
def get_scores():
    pass



