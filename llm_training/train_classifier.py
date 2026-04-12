# train_classifier.py

import torch
import time

# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#     logits = model(input_batch)[:, -1, :]  # Logits of last output token
#     loss = torch.nn.functional.cross_entropy(logits, target_batch)
#     return loss

def calc_loss_batch(input_batch, target_batch, model, device, pad_token_id=50256):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    ### >>> CHANGED: use last non-PAD token instead of -1
    logits_all = model(input_batch)  # [B, T, C]

    non_pad_mask = (input_batch != pad_token_id)
    last_idx = non_pad_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_batch.size(0), device=device)

    logits = logits_all[batch_indices, last_idx, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_loss, count = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            count += 1

    return total_loss / count

def calc_accuracy_loader(data_loader, model, device, num_batches=None, pad_token_id=50256):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Accuracy evaluation does NOT require gradients
    # Without torch.no_grad(), PyTorch builds a computation graph. This graph tracks all operations on tensors that require gradients, which is essential for backpropagation during training. However, during evaluation, we don't need to compute gradients or build this graph. Using torch.no_grad() tells PyTorch not to track operations for gradient computation, which reduces memory usage and speeds up computations since it doesn't have to maintain the graph.
    # Using torch.no_grad() makes evaluation MUCH faster
    # It disables: gradient tracking, autograd graph creation,unnecessary memory allocation
    # This gives you: lower memory usage, faster inference, more stable evaluation
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            logits_all = model(input_batch)
            non_pad_mask = (input_batch != pad_token_id)
            last_idx = non_pad_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_batch.size(0), device=device)
            logits = logits_all[batch_indices, last_idx, :]

            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples

# def calc_accuracy_loader(data_loader, model, device, num_batches=None):
#     model.eval()
#     correct_predictions, num_examples = 0, 0

#     if num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))

#     # Accuracy evaluation does NOT require gradients
#     # Without torch.no_grad(), PyTorch builds a computation graph. This graph tracks all operations on tensors that require gradients, which is essential for backpropagation during training. However, during evaluation, we don't need to compute gradients or build this graph. Using torch.no_grad() tells PyTorch not to track operations for gradient computation, which reduces memory usage and speeds up computations since it doesn't have to maintain the graph.
#     # Using torch.no_grad() makes evaluation MUCH faster
#     # It disables: gradient tracking, autograd graph creation,unnecessary memory allocation
#     # This gives you: lower memory usage, faster inference, more stable evaluation
#     with torch.no_grad():
#         for i, (input_batch, target_batch) in enumerate(data_loader):
#             if i >= num_batches:
#                 break

#             input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#             logits = model(input_batch)[:, -1, :]
#             predicted_labels = torch.argmax(logits, dim=-1)

#             num_examples += predicted_labels.shape[0]
#             correct_predictions += (predicted_labels == target_batch).sum().item()

#     return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        # below call sets the model into training mode
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()   # 1. clear old gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()         # 2. compute new gradients
            optimizer.step()        # 3. update weights

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Epoch accuracy
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | "
              f"Validation accuracy: {val_accuracy*100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen



# ✔ Your dataset = a book
# ✔ A batch = a few pages (so the model can learn from multiple examples at once and this is defined by the user based on the memory and speed)
# ✔ An epoch = reading the whole book once
# ✔ Multiple epochs = reading the book multiple times to learn better
# So:
# 1 epoch → the model sees every training sample once
# 5 epochs → the model sees every training sample five times
# 10 epochs → ten times, and so on
