import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable
from logging import Logger
from nn.chebnet import ChebNet

def test(testloader: DataLoader,
         model: torch.nn.Module,
         loss_fn: Callable,
         logger: Logger,
         device: torch.device,
         ) -> tuple[float, float]:
    model.eval()
    # test_loss = 0
    test_loss = []
    test_correct = 0
    total_samples = 0
    preds = []
    end = time.time()
    with torch.no_grad():
        for input, label, node, adj, lmax in testloader:

            input, label = input.to(device), label.to(device)
            adj = adj.to(device)
            node = node.to(device)
            lambda_max = lmax.to(device)

            if isinstance(model, ChebNet):
                output = model(adj, node, input, lambda_max)
            else:
                output = model(adj, node, input)
            # test_loss += loss_fn(output, label).item() * input.shape[0]
            test_loss += [loss_fn(output, label).item()] * input.shape[0]

            pred = output.argmax(1, keepdim=True)
            if (pred.numel() == 1):
                preds.append(pred.item())
            else:
                preds.extend(pred.squeeze().tolist())
            test_correct += pred.eq(label.view_as(pred)).sum().item()
            total_samples += input.shape[0]

    # test_loss /= total_samples
    test_loss = np.array(test_loss)
    avg_loss = (test_loss / total_samples).sum()
    test_accuracy = 100. * test_correct / total_samples
    infer_time = time.time() - end
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) "\
                "Inference time: {:.4f}s".format(
        avg_loss, test_correct, total_samples, test_accuracy, infer_time))
    return avg_loss, test_accuracy, preds