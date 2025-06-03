import torch
from torch import nn

class Trainer:
    def __init__(self, model, device = torch.device('mps' if torch.mps.is_available() else 'cpu'), alpha = 0.5, temperature = 3):
        self.model = model.to(device)
        self.alpha = alpha
        self.temperature = temperature
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0
        self.device = device

    def init_params(self):
        # Reset training and validation statistics
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0

    def train_step(self, data, criterion, optimizer):
        # Perform a single training step (forward + backward pass)
        self.model.train()
        X, y = data[0].to(self.device), data[1].to(self.device)
        optimizer.zero_grad()
        y_pred = self.model(X)
        # Compute loss
        loss = criterion(y_pred, y)
        self.train_loss += loss.item()
        # Get predicted class
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.train_acc += pred.eq(y.view_as(pred)).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()

    def train_step_kd(self, data, teacher_logits, criterion, optimizer):
        # Perform a single training step (forward + backward pass)
        self.model.train()
        X, y = data[0].to(self.device), data[1].to(self.device)
        teacher_logits = teacher_logits.to(self.device)
        optimizer.zero_grad()
        y_pred = self.model(X)
        # Compute loss
        loss = criterion(y_pred, teacher_logits, y, self.alpha, self.temperature)
        self.train_loss += loss.item()
        # Get predicted class
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.train_acc += pred.eq(y.view_as(pred)).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()

    def val_step(self, data, criterion):
        # Perform a validation step
        self.model.eval()
        X, y = data[0].to(self.device), data[1].to(self.device)
        y_pred = self.model(X)
        # Compute loss
        loss = criterion(y_pred, y)
        self.val_loss += loss.item()
        # Get predicted class
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.val_acc += pred.eq(y.view_as(pred)).sum().item()

def distillation_loss(student_logits, teacher_logits, temperature = 3.0):
    """
    Calculates the distillation loss between the student and teacher models.

    student_logits: raw predictions (logits) from the student model (before softmax)
    teacher_logits: raw predictions (logits) from the teacher model (before softmax)
    temperature: scaling factor to soften probability distributions
    """
    # Apply softmax with temperature to get probability distributions
    student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)

    # Compute KL-Divergence (distillation loss)
    loss = nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss

def total_loss(student_logits, teacher_logits, labels, alpha = 0.5, temperature = 3.0):
    """
    Computes the total loss for training the student model.
    student_logits: raw predictions from the student model
    teacher_logits: raw predictions from the teacher model
    labels: ground truth labels
    alpha: weight factor to balance cross-entropy and distillation loss
    temperature: scaling factor for distillation
    """
    # Compute standard cross-entropy loss for classification
    ce_loss = nn.functional.cross_entropy(student_logits, labels)
    # Compute distillation loss using teacher's guidance
    dist_loss = distillation_loss(student_logits, teacher_logits, temperature)
    # Combine both losses with weight alpha
    return alpha * ce_loss + (1 - alpha) * dist_loss