import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable
from logging import Logger
from nn.chebnet import ChebNet

def train(trainloader: DataLoader,
          model: torch.nn.Module,
          loss_fn: Callable,
          optimizer: torch.optim.Optimizer,
          epoch: int,
          logger: Logger,
          device: torch.device,
          ) -> tuple[float, float, list[int]]:
    model.train()
    train_correct = 0
    total_samples = 0
    end = time.time()
    log_step = int(len(trainloader) / 2)
    for i, (input, label, node, adj, lmax) in enumerate(trainloader):

        input, label = input.to(device), label.to(device)
        adj = adj.to(device)
        node = node.to(device)
        lambda_max = lmax.to(device)
        data_time = time.time() - end
        end = time.time()

        if isinstance(model, ChebNet):
            output = model(adj, node, input, lambda_max)
        else:
            output: torch.Tensor = model(adj, node, input)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000, 2)
        optimizer.step()
        train_time = time.time() - end
        # pred = output.argmax(dim=1, keepdim=True)
        # train_correct += pred.eq(label.view_as(pred)).sum().item()
        # train_loss += loss.item() * input.shape[0]
        total_samples += input.shape[0]

        if (log_step != 0) and (i % log_step == 0):
            logger.info( "Train Epoch: {} [{}/{} ({:.0f}%)] "\
                         "Loss: {:.6f} Data loading time: {:.4f}s "\
                         "Train time: {:.4f}s".format(
                epoch, total_samples, len(trainloader.dataset),
                100. * total_samples / len(trainloader.dataset), loss.item(),
                data_time, train_time))
        
        end = time.time()

    # train_loss = 0
    train_loss = []
    total_samples = 0
    preds = []
    model.eval()
    with torch.no_grad():
        for i, (input, label, node, adj, lmax) in enumerate(trainloader):

            input, label = input.to(device), label.to(device)
            adj = adj.to(device)
            node = node.to(device)
            lambda_max = lmax.to(device)

            if isinstance(model, ChebNet):
                output = model(adj, node, input, lambda_max)
            else:
                output = model(adj, node, input)
            loss = loss_fn(output, label)
            pred = output.argmax(dim=1, keepdim=True)
            if (pred.numel() == 1):
                preds.append(pred.item())
            else:
                preds.extend(pred.squeeze().tolist())
            train_correct += pred.eq(label.view_as(pred)).sum().item()
            # train_loss += loss.item() * input.shape[0]
            train_loss += [loss.item()] * input.shape[0]
            total_samples += input.shape[0]

    # train_loss /= total_samples
    train_loss = np.array(train_loss)
    avg_loss = (train_loss / total_samples).sum()
    train_accuracy = 100. * train_correct / total_samples
    infer_time = time.time() - end
    logger.info("Train set: Average loss: {:.4f}, "\
                "Accuracy: {}/{} ({:.2f}%) Inference time: {:.4f}s".format(
        avg_loss, train_correct, total_samples, train_accuracy, infer_time))
    return avg_loss, train_accuracy, preds