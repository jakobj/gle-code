import torch
import numpy as np

from torch.utils.data import DataLoader
from lib.visualization import plot_sliding_outputs
from sklearn.metrics import confusion_matrix


def train(params, MemoryClass, model, loss_fn, train_loader, optimizer, epoch, fn_out, use_le=True, optimizer_step_interval=1):
    model.train()
    n_correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_loss = 0

        if params['use_cuda']:
            data, target = data.cuda(), target.cuda()

        # implement sliding window and sum logits over windows
        memory = MemoryClass(data, **MemoryClass.kwargs)
        n_steps = len(memory)

        output_over_time = []
        for step, input in enumerate(memory):
            optimizer.zero_grad()
            if use_le:
                with torch.no_grad():
                    # calling the model automatically populates the gradients
                    output = model(input, target, beta=params['beta'])
                    loss = loss_fn(output, target, reduction='sum')
            else:
                output = model(input)
                loss = loss_fn(output, target, reduction='sum')
                loss.backward()

            output_over_time.append(output.detach())

            for p in model.parameters():
                assert torch.all(torch.isfinite(p.grad)), breakpoint()
            if step % optimizer_step_interval == 0:
                optimizer.step()
            for p in model.parameters():
                assert torch.all(torch.isfinite(p)), breakpoint()
            optimizer.zero_grad()

            # average loss over steps for individual batch
            batch_loss += loss.item() / n_steps / input.shape[0]

            # average loss over steps of whole train set
            train_loss += loss.item() / n_steps

        output_over_time = torch.stack(output_over_time, dim=1)  # (batch_size, n_steps, output_size)
        pred = output_over_time.sum(axis=1).argmax(dim=1, keepdim=True)
        n_correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % params['log_interval'] == 0:
            print('Train Epoch: {}({}) [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss))

        if fn_out is not None and batch_idx % params['checkpoint_interval'] == 0:
            torch.save(model.state_dict(), fn_out.format(postfix=f'_{epoch}_{batch_idx}'))

    # average loss over dataset
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * n_correct / len(train_loader.dataset)

    print('\nTrain: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, n_correct, len(train_loader.dataset), train_acc))

    return train_loss, train_acc

def test(params, MemoryClass, model, loss_fn, test_loader, epoch=0, prefix='valid', lr_scheduler=None):
    model.eval()
    n_correct = 0
    test_loss = 0

    # collect for confusion matrix
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if params['use_cuda']:
                data, target = data.cuda(), target.cuda()

            memory = MemoryClass(data, **MemoryClass.kwargs)
            n_steps = len(memory)

            output_over_time = []
            for input in memory:
                output = model(input)
                test_loss += loss_fn(output, target, reduction='sum').item() / n_steps
                output_over_time.append(output.detach())

            output_over_time = torch.stack(output_over_time, dim=1)  # (batch_size, n_steps, output_size)
            pred = output_over_time.sum(axis=1).argmax(dim=1, keepdim=True)
            n_correct += pred.eq(target.view_as(pred)).sum().item()

            preds_list.append(pred.detach().cpu().numpy())
            targets_list.append(target.view_as(pred).detach().cpu().numpy())

    # average loss over dataset
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * n_correct / len(test_loader.dataset)

    if lr_scheduler is not None and epoch > 0:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step()
        elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(test_loss)
        else:
            raise NotImplementedError()
        print('Using learning rate:', lr_scheduler.optimizer.param_groups[0]["lr"])

    print('Evaluate on', prefix, 'set: Average loss per sample: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, n_correct, len(test_loader.dataset), test_acc))

    # plot confusion matrix to stdout (visible in neptune.ai under monitoring/stdout)
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    print("Confusion matrix: ")
    print(confusion_matrix(preds, targets))

    return test_loss, test_acc

def mnist1d_run(params, memory, model, loss_fn, fn_out, train_data, test_data, optimizer=None, lr_scheduler=None, use_le=True, optimizer_step_interval=1):

    metrics = {metric: [] for metric in ['train_loss', 'train_acc', 'test_loss', 'test_acc', 'last_lr']}

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size_test'], shuffle=False)

    memory_type = memory

    if optimizer is None:
        assert False
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    test_loss, test_acc = test(params, memory_type, model, loss_fn, test_loader, epoch=0, prefix='valid', lr_scheduler=lr_scheduler)

    if 'test_loss' in metrics:
        # NOTE wrong first value, necessary to make lists equally long and thus
        # pandas happy
        metrics['train_loss'].append(test_loss)
        metrics['test_loss'].append(test_loss)
    if 'test_acc' in metrics:
        metrics['train_acc'].append(test_acc)  # NOTE wrong first value
        metrics['test_acc'].append(test_acc)
    if 'last_lr' in metrics:
        metrics['last_lr'].append(lr_scheduler.optimizer.param_groups[0]["lr"])

    print('Start training:')
    for epoch in range(1, params['epochs'] + 1):
        train_loss, train_acc = train(params, memory_type, model, loss_fn, train_loader, optimizer, epoch, fn_out, use_le=use_le, optimizer_step_interval=optimizer_step_interval)
        if 'train_loss' in metrics:
            metrics['train_loss'].append(train_loss)
        if 'train_acc' in metrics:
            metrics['train_acc'].append(train_acc)
        test_loss, test_acc = test(params, memory_type, model, loss_fn, test_loader, epoch=epoch, prefix='valid', lr_scheduler=lr_scheduler)
        if 'test_loss' in metrics:
            metrics['test_loss'].append(test_loss)
        if 'test_acc' in metrics:
            metrics['test_acc'].append(test_acc)
        if 'last_lr' in metrics:
            metrics['last_lr'].append(lr_scheduler.optimizer.param_groups[0]["lr"])

    if fn_out is not None:
        torch.save(model.state_dict(), fn_out.format(postfix=''))

    return metrics
