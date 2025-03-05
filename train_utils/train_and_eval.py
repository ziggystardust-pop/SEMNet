import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import numpy as np


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            dice_loss_value = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            loss = loss+dice_loss_value
            losses[name] = loss
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']



def evaluate(model, data_loader, device, num_classes,ds=0):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_loss = 0.0
    num_batches = 0
    if ds == 0:
        with torch.no_grad():
            for image, target in metric_logger.log_every(data_loader, 100, header):
                image, target = image.to(device), target.to(device)
                output = model(image)
                out = output['out']
                loss = criterion(output, target, num_classes=num_classes, ignore_index=255)
                total_loss += loss.item()
                num_batches += 1
    
                confmat.update(target.flatten(), out.argmax(1).flatten())
                dice.update(out, target)
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Test Loss: {avg_loss}")
    
            confmat.reduce_from_all_processes()
            dice.reduce_from_all_processes()
    else:
        with torch.no_grad():
            for image, target in metric_logger.log_every(data_loader, 100, header):
                image, target = image.to(device), target.to(device)
                output = model(image)
                out = output[0]
                loss_main = criterion({"out":output[0]}, target,num_classes=num_classes, ignore_index=255)
                loss_5 = criterion({"out":output[1]}, target,num_classes=num_classes, ignore_index=255)
                loss_4 = criterion({"out":output[2]}, target,num_classes=num_classes, ignore_index=255)
                loss_3 = criterion({"out":output[3]}, target,num_classes=num_classes, ignore_index=255)
                loss = loss_main + 0.4 * loss_5 + 0.3 * loss_4 + 0.2 * loss_3
                total_loss += loss.item()
                num_batches += 1
    
                confmat.update(target.flatten(), out.argmax(1).flatten())
                dice.update(out, target)
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Test Loss: {avg_loss}")
    
            confmat.reduce_from_all_processes()
            dice.reduce_from_all_processes()

    return confmat, dice.value.item(),avg_loss



def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None,ds=0,dataname = "benx"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:

        loss_weight = None


    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)

            if ds == 1:
                loss_main = criterion({"out":output[0]}, target, loss_weight, num_classes=num_classes, ignore_index=255)
                loss_5 = criterion({"out":output[1]}, target, loss_weight, num_classes=num_classes, ignore_index=255)
                loss_4 = criterion({"out":output[2]}, target, loss_weight, num_classes=num_classes, ignore_index=255)
                loss_3 = criterion({"out":output[3]}, target, loss_weight, num_classes=num_classes, ignore_index=255)
                loss = loss_main + 0.4 * loss_5 + 0.3 * loss_4 + 0.2 * loss_3

            else:
                loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
