import os
import shutil
import re
import time

import numpy as np
import oneflow as flow


class Solver(object):

    def __init__(self, data, model, optimizer, device, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.device = device
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = []
        self.cv_loss = []


        self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch+1))
                flow.save(self.model.state_dict(), file_path)
                print('Saving checkpoint model to %s' % file_path)
                for dirs in os.listdir(self.save_folder):
                    dir_name = os.path.join(self.save_folder, dirs)
                    dir = dir_name.split('/')[-1]
                    dir = re.findall(r"\d+", dir)
                    if dir == []:
                        dir = 1000
                    else:
                        dir = int(dir[0])
                    if (epoch+1) - dir >= 5:
                        shutil.rmtree(dir_name)

            # Cross validation
            print('Cross validation...')
            self.model.eval()
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr and val_loss >= self.prev_val_loss:
                if self.early_stop and self.halving:
                    print("Already start halving learing rate, it still gets "
                          "too small imporvement, stop training early.")
                    break
                self.halving = True
            if self.halving:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2.0
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=param_group['lr']))
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss.append(tr_avg_loss)
            self.cv_loss.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                if os.path.exists(file_path):
                    shutil.rmtree(file_path)
                flow.save(self.model.state_dict(), file_path)
                np.save(file_path+'/tr_loss.npy', self.tr_loss)
                np.save(file_path+'/val_loss.npy', self.cv_loss)
                np.save(file_path + '/epoch.npy', epoch)

                print("Find better validated model, saving to %s" % file_path)
            
                
            
    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            start1 = time.time()
            padded_input, input_lengths, padded_target = data
            padded_input = padded_input.to(self.device)
            input_lengths = input_lengths.to(self.device)
            padded_target = padded_target.to(self.device)
            start2 = time.time()
            loss = self.model(padded_input, input_lengths, padded_target)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = flow.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                        self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            print('data-time3', 1000 * (time.time() - start2))
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                    'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)

        return total_loss / (i + 1)
