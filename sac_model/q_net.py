from torch import nn


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=300, h2=400):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(action_space.shape[0] + h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=-1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)

