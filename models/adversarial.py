import torch
import torch.nn as nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x) # Avoid modifying the input tensor directly

    @staticmethod
    def backward(ctx, grad_output):
        # Return the negative gradient multiplied by alpha, and None for alpha's gradient
        return grad_output.neg() * ctx.alpha, None
        
class AdversarialClassifier(nn.Module):
    def __init__(self, feature_extractor, embedding_dim, num_classes, num_centers, lambda_=1):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ) # Classification head
        self.grl = GradientReversalLayer
        self.center_fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_centers))
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
    
    def forward(self, x):
        features = self.feature_extractor(x).last_hidden_state[:, 0, :]
        class_pred = self.fc(features)
        rev_x = self.grl.apply(features, self.lambda_)  # Reverse gradients for adversarial learning
        center_pred = self.center_fc(rev_x)
        return class_pred, center_pred