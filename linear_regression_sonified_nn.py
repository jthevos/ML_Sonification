from synthesizer import Player, Synthesizer, Waveform
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Train the model using gradient descent

As seen above, we reduce the loss and improve our model using the gradient descent optimization algorithm. Thus, we can train the model using the following steps:

    Generate predictions

    Calculate the loss

    Compute gradients w.r.t the weights and biases

    Adjust the weights by subtracting a small quantity proportional to the gradient

    Reset the gradients to zero

"""
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# Define dataset
train_ds = TensorDataset(inputs, targets)

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# shuffle will shuffle first and THEN batch


# Define model

# remember, 3 inputs (temp, rainfall, humidity), 2 outputs (apples, oranges)
# the nn.Linear(rows, cols) function will automatically initialize our weights and biases
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

# generate predictions
preds = model(inputs)

# Define loss function
loss_fn = F.mse_loss

loss = loss_fn(model(inputs), targets)

# Define optimizer

# remember the 1e-5 is a hyper parameter, lr stands for learning rate.

# we need to pass model.parameters() here because the optimizer needs to know which matrices should
# be modified during the update step. In this case, it is both of our parameters.
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


def mapValue(value, minValue, maxValue, minResultValue, maxResultValue):
    """
    Maps value from a given source range, i.e., (minValue, maxValue),
    to a new destination range, i.e., (minResultValue, maxResultValue).
    The result will be converted to the result data type (int, or float).
    """
    # check if value is within the specified range
    if value < minValue or value > maxValue:
      raise ValueError("value, " + str(value) + ", is outside the specified range, " \
                                 + str(minValue) + " to " + str(maxValue) + ".")

    # we are OK, so let's map
    value = float(value)  # ensure we are using float (for accuracy)
    normal = (value - minValue) / (maxValue - minValue)   # normalize source value

    # map to destination range
    result = normal * (maxResultValue - minResultValue) + minResultValue

    destinationType = type(minResultValue)  # find expected result data type
    result = destinationType(result)        # and apply it

    return result

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):

    player = Player()
    player.open_stream()
    synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
    # Play A4

    max_loss = 0   # initialize max_loss to zero. The first iteration will set this to the worst loss

    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb,yb in train_dl:

            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss between the predictions and the targets
            loss = loss_fn(pred, yb)

            # Because we are using a linear mapping, we need to set a relative max
            # We don't know how lossy the first epoch will be because the predictions
            # are randomly generated for the first run. We want the lossiest predictions
            # to sound furthest away from the targets.
            if loss.item() > max_loss:
                max_loss = loss.item()
                print("max_loss", max_loss)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch+1) % 10 == 0:
            # loss.item gets the actual loss out of the loss tensor.
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


            log_loss = np.log(loss.item())   # use a log of the true loss in order to hear a smoother descent
            relative_max = np.log(max_loss)  # get a relative maximum by taking the log of the max_loss

            # map the log of the loss, from the 0 - relative_max range
            # to the specified range, in this case, two full octaves
            loss_frequency = mapValue(log_loss, 0, relative_max, 220.0, 880.0)

            # print for debug
            print("loss.item()", loss.item(), "loss_frequency", loss_frequency)

            # play the sound
            player.play_wave(synthesizer.generate_chord([220.0, loss_frequency], 1.0))


fit(500, model, loss_fn, opt, train_dl)
