import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from BayesLayers import log_gaussian, BayesMLP
from utils import printProgressBar
import pickle
import os

parameters = {
    'batch_size': 100,
    'prior_sigma': np.exp(-3),
    'n_samples': 3,
    'learning_rate': 0.001,
    'n_epochs': 100,
    'num_hidden_units': 1200,
    'dataset': 'MNIST'
}

output_dir = f'hidden_{parameters["num_hidden_units"]}__priorsigma_{parameters["prior_sigma"]:{2}.{2}}__nsamples{parameters["n_samples"]}/'
output_dir = 'output/' + output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir + 'params.txt', 'w') as f:
    for param, value in parameters.items():
        f.write(f'{param}: {value}\n')

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = datasets.MNIST('./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('./data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=parameters['batch_size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False)

n_batchs = len(train_loader)
input_shape = train_set[0][0].size()

model = BayesMLP(input_shape[0] * input_shape[1] * input_shape[2], 10, parameters['prior_sigma'],
                 parameters['num_hidden_units'])
model.cuda()

optimiser = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])


def loss_func(log_qw, log_pw, log_likelihood):
    return ((1. / n_batchs) * (log_qw - log_pw) - log_likelihood).sum()


def evaluate_model(model, dataloader):
    count, correct = 0, 0

    for idx, (x, y) in enumerate(dataloader):
        pred = model(x.cuda(), fix_weights=True)
        _, pred = torch.max(pred, 1)
        correct += np.count_nonzero(pred.data.cpu().numpy() == y.data.cpu().numpy())
        count += y.size()[0]

    return correct / count


# def save_checkpoint(state, filename):


plt.ion()
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
test_accuracy_plot, = ax1.plot(1, 1)
train_accuracy_plot, = ax1.plot(1, 1)
loss_plot, = ax2.plot(1, 1)
fig.canvas.draw()
plt.show(block=False)

test_accs = []
train_accs = []
losses = []

for e in range(parameters['n_epochs']):
    cumulative_loss = 0
    for idx, (x, y) in enumerate(train_loader):
        printProgressBar(idx, len(train_loader), f'Epoch {e}')
        model.zero_grad()

        log_qw, log_pw, log_likelihood = model.bayes_sample_loss(x, y, parameters['n_samples']) #= 0., 0., 0.
        #
        # for _ in range(parameters['n_samples']):
        #     y_hat = model(x.cuda())
        #
        #     rows = torch.LongTensor(np.arange(0, y.size()[0]))
        #     i = torch.stack([rows, y], 0)
        #     y_ohe = torch.sparse.FloatTensor(i, torch.ones(y.size()[0])).to_dense().cuda()
        #
        #     sample_log_qw, sample_log_pw = model.get_losses()
        #     sample_log_likelihood = log_gaussian(y_ohe, y_hat, (torch.ones(
        #         y_hat.size()) * parameters['prior_sigma']).cuda()).sum()  # -F.nll_loss(torch.log(y_hat), y.cuda(), reduction='sum')
        #
        #     log_qw += sample_log_qw / parameters['n_samples']
        #     log_pw += sample_log_pw / parameters['n_samples']
        #     log_likelihood += sample_log_likelihood / parameters['n_samples']

        loss = loss_func(log_qw, log_pw, log_likelihood)
        cumulative_loss += loss
        loss.backward()
        optimiser.step()

    train_accuracy = evaluate_model(model, train_loader)
    test_accuracy = evaluate_model(model, test_loader)

    losses.append(cumulative_loss.detach().cpu().numpy().tolist())
    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)

    x = list(range(1, len(train_accs) + 1))

    train_accuracy_plot.set_xdata(x)
    train_accuracy_plot.set_ydata(train_accs)
    test_accuracy_plot.set_xdata(x)
    test_accuracy_plot.set_ydata(test_accs)
    loss_plot.set_xdata(x)
    loss_plot.set_ydata(losses)

    ax1.relim()
    ax1.autoscale_view(True, True, True)
    ax2.relim()
    ax2.autoscale_view(True, True, True)

    fig.canvas.draw()

    fig.savefig(output_dir + 'metrics_plot.')

    data = {
        'train_accuracies': train_accs,
        'test_accuracies': test_accs,
        'losses': losses
    }

    data = pd.DataFrame(data)
    data.to_csv(output_dir + 'metrics.csv')

    torch.save({'epoch':e, 'state_dict': model.state_dict(), 'test_accuracy': test_accuracy, 'train_accuracy': train_accuracy}, output_dir+'latest.pt')

    if sum(i > test_accuracy for i in test_accs) == 0:  # if best test accuracy seen so far
        torch.save({'epoch': e, 'state_dict': model.state_dict(), 'test_accuracy': test_accuracy, 'train_accuracy': train_accuracy}, output_dir + 'best_test.pt')

    print(
        f'\rEpoch {e} Loss: {loss:{2}.{12}}, Train accuracy: {train_accuracy:{2}.{4}}, Test accuracy: {test_accuracy:{2}.{4}}                                          ')
