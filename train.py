import os
import torch
import numpy as np
from tqdm import tqdm

from utils import *
    
def train(args):
    configs = initialize(args)
    
    model = configs['model']
    criterion = configs['criterion']
    optimizer = configs['optimizer']
    converter = configs['converter']
    scheduler = configs['scheduler']
    out_log = configs['out_log']

    best_cer = 1e5

    for epoch in range(args.num_epochs):
        total_steps = len(configs['train_loader'])
        print(total_steps)
        current_step = 0
        
        # Training
        for images, labels in tqdm(configs['train_loader'], total=total_steps):
            current_step += 1
            batch_size = images.size(0)
            text, length = converter.encode(labels, batch_max_length=args.max_length)

            model.train()
            preds, visual_feature = model(images.to(device), text[:, :-1], True) # align with Attention forward
            targets = text[:, 1:] # remove [GO] symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
            
            model.zero_grad()
            cost.backward()
            optimizer.step()
            
            # Log step
            if current_step % args.log_every == 0:
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Train Loss {cost.detach().cpu().mean()}"
                write_train_log(log_message, out_log, 'a')
                print(log_message)
                
            # Val step
            if current_step % args.val_every == 0:
                model.eval()
                loss_avg = []
                val_cer = []
                length_for_pred = torch.IntTensor([args.max_length] * batch_size)
                text_for_pred = torch.LongTensor(batch_size, args.max_length + 1).fill_(0)

                for images, labels in tqdm(configs['test_loader'], total=len(configs['test_loader'])):
                    text, length = converter.encode(labels, batch_max_length=args.max_length)
                    
                    with torch.no_grad():
                        preds, visual_feature = model(images.to(device), text[:, :-1], True) # Align with Attention
                    target = text[:, 1:]
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    loss_avg.append(cost.detach().cpu())
                    
                    try:
                        _, preds_index = preds.detach().cpu().max(2)
                        preds_str = converter.decode(preds_index, length_for_pred)
                        labels = converter.decode(target, length)
                        val_cer.append(calc_cer(preds_str, labels, batch_size))
                    except:
                        pass
                    
                val_cer = np.mean(np.array(val_cer))
                    
                loss_avg = torch.stack(loss_avg, 0)
                loss_avg = loss_avg.cpu().mean()
                
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Validation Loss {loss_avg} | CER {val_cer}"
                write_train_log(log_message, out_log, 'a')
                print(log_message)
                
                # Best cer
                if val_cer < best_cer:
                    best_cer = val_cer
                    model.train()
                    print(f"[INFO] Saving model with best CER {best_cer}")
                    torch.save(model.state_dict(), os.path.join(args.out_dir, f"best_{best_cer}.pt"))
                
        # if (epoch+1) % args.save_every == 0:
        #     model.train()
        #     print(f"[INFO] Saving model at epoch {epoch+1}")
        #     torch.save(model.state_dict(), os.path.join(args.out_dir, f"E_{epoch+1}.pth"))
            
            scheduler.step()