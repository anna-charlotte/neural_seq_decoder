import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_features: int, conditioning_dim: int, n_layers: int):
        super(FiLM, self).__init__()

        if n_layers not in [1, 2]:
            raise ValueError(f"n_layers for FiLM must be either 1 or 2, given: {n_layers}")
        
        if n_layers == 1:
            self.scale_network = nn.Sequential(
                nn.Linear(conditioning_dim, in_features)
            )
            self.shift_network = nn.Sequential(
                nn.Linear(conditioning_dim, in_features)
            )
        elif n_layers == 2:
            self.scale_network = nn.Sequential(
                nn.Linear(conditioning_dim, int(in_features / 2)),
                nn.ReLU(),
                nn.Linear(int(in_features / 2), in_features)
            )
            self.shift_network = nn.Sequential(
                nn.Linear(conditioning_dim, int(in_features / 2)),
                nn.ReLU(),
                nn.Linear(int(in_features / 2), in_features)
            )
    
    def forward(self, x, c):
        gamma = self.scale_network(c)
        beta = self.shift_network(c)

        gamma = gamma.reshape(x.shape)
        beta = beta.reshape(x.shape)

        return gamma * x + beta



class __FiLM_2(nn.Module):
    def __init__(self, in_features: int, conditioning_dim: int):
        super(FiLM_2, self).__init__()

        self.scale_network = nn.Sequential(
            nn.Linear(conditioning_dim, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), in_features)
        )
        self.shift_network = nn.Sequential(
            nn.Linear(conditioning_dim, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), in_features)
        )
    
    def forward(self, x, c):
        gamma = self.scale_network(c)
        beta = self.shift_network(c)
        if len(x.size()) > 2:
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)


        return gamma * x + beta


class __FiLM(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(emb_dim, in_channels)
        self.shift = nn.Linear(emb_dim, in_channels)

    def forward(self, x, y_emb):
        scale = self.scale(y_emb).unsqueeze(2).unsqueeze(3)
        shift = self.shift(y_emb).unsqueeze(2).unsqueeze(3)
        return x * scale + shift