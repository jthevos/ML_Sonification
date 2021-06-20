from synthesizer import Player, Synthesizer, Waveform

import numpy as np
import torch

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


# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# confirm successful conversion
print(inputs)
print(targets)


# Weights and biases

# random, two rows, three columns
# randn usually comes from guassian distribution
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)


# we need w.t() because we need to transpose the weights for matrix mult to work
def model(x):
    return x @ w.t() + b

# Generate predictions from randomly generated weights and biases
preds = model(inputs)
print(preds)

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# Compute loss
loss = mse(preds, targets)
print(loss)

# # Compute gradients
# loss.backward()
#
# # Gradients for weights
# print(w)
# print(w.grad)
# #
# with torch.no_grad():
#     w -= w.grad * 1e-5
#     b -= b.grad * 1e-5
#
# # Let's verify that the loss is actually lower
# loss = mse(preds, targets)
# print(loss)
#
# # reset gradients to 0
# w.grad.zero_()
# b.grad.zero_()
# print(w.grad)
# print(b.grad)

"""
Train the model using gradient descent

As seen above, we reduce the loss and improve our model using the gradient descent optimization algorithm. Thus, we can train the model using the following steps:

    Generate predictions

    Calculate the loss

    Compute gradients w.r.t the weights and biases

    Adjust the weights by subtracting a small quantity proportional to the gradient

    Reset the gradients to zero

"""

loss = mse(preds, targets)
print(loss)

# Compute gradients
loss.backward()
print(w.grad)
print(b.grad)


# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# Train for 500 epochs
for i in range(500):
    # get predictions by passing inputs into the defined model
    preds = model(inputs)

    # get loss by getting the mean squared error of the predictions and the targets
    loss = mse(preds, targets)

    # get gradients (derivatives) with .backward()
    loss.backward()

    # now, tell pytorch to stop tracking gradients for a second, and loop
    with torch.no_grad():

        # adjust weights by a small proportion of the weight gradient (the 1e-5 is a hyper param, adjust with orders of magnitide)
        w -= w.grad * 1e-5

        # adjust bias by a small proportion of the bias gradient (the 1e-5 is a hyper param)
        b -= b.grad * 1e-5

        # zero out both gradients because they accumulate!!
        w.grad.zero_()
        b.grad.zero_()


# player = Player()
# player.open_stream()
# synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
# # Play A4
# player.play_wave(synthesizer.generate_constant_wave(440.0, 3.0))
