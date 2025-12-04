import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class NeuralNet(nn.Module):
    def __init__(self, dim=2, hidden_dim=64, n_hidden=2, torso=None, output_dim=3):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        if torso is not None:
            self.torso = torso
        else:
            self.torso = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ELU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    for _ in range(n_hidden)
                ],
            )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.output_layer(self.torso(x))
        return out


class CirclePF(NeuralNet):
    def __init__(
        self,
        hidden_dim=64,
        n_hidden=2,
        beta_min=0.1,
        beta_max=2.0,
    ):
        output_dim = 3 # Only alpha and beta for single Beta distribution
        super().__init__(
            dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim
        )

        # The following parameters are for PF(. | s0)
        self.PFs0 = nn.ParameterDict(
            {
                "log_alpha_r": nn.Parameter(torch.zeros(1)),
                "log_alpha_theta": nn.Parameter(torch.zeros(1)),
                "log_beta_r": nn.Parameter(torch.zeros(1)),
                "log_beta_theta": nn.Parameter(torch.zeros(1)),
            }
        )

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        out = super().forward(x)
        pre_sigmoid_exit = out[..., 0]
        log_alpha = out[..., 1]
        log_beta = out[..., 2]

        exit_proba = torch.sigmoid(pre_sigmoid_exit)
        return (
            exit_proba,
            self.beta_max * torch.sigmoid(log_alpha) + self.beta_min,
            self.beta_max * torch.sigmoid(log_beta) + self.beta_min,
        )

    def to_dist(self, x):
        if torch.all(x[0] == 0.0):
            assert torch.all(
                x == 0.0
            )  # If one of the states is s0, all of them must be
            alpha_r = self.PFs0["log_alpha_r"]
            alpha_r = self.beta_max * torch.sigmoid(alpha_r) + self.beta_min
            alpha_theta = self.PFs0["log_alpha_theta"]
            alpha_theta = self.beta_max * torch.sigmoid(alpha_theta) + self.beta_min
            beta_r = self.PFs0["log_beta_r"]
            beta_r = self.beta_max * torch.sigmoid(beta_r) + self.beta_min
            beta_theta = self.PFs0["log_beta_theta"]
            beta_theta = self.beta_max * torch.sigmoid(beta_theta) + self.beta_min

            dist_r = Beta(alpha_r[0], beta_r[0])
            dist_theta = Beta(alpha_theta[0], beta_theta[0])
            return dist_r, dist_theta

        # Otherwise, we use the neural network
        exit_proba, alpha, beta = self.forward(x)
        dist = Beta(alpha, beta)

        return exit_proba, dist

class Uniform:
    def __init__(self, delta=0.25, box_max=1.0):
        """
        delta: env.delta (한 step의 최대 길이)
        box_max: [0, box_max]×[0, box_max] 영역을 가정
        """
        self.delta = float(delta)
        self.box_max = float(box_max)
        # 예: box_max=1.0, delta=0.25 -> L = 4
        self.L = int(round(self.box_max / self.delta))

    def _level(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, dim>=2)
        level k(x) = floor(max(x, y) / delta) in {0, ..., L}
        """
        # 2D라고 가정 (필요하면 x[:, :2]만 사용)
        max_coord, _ = x[:, :2].max(dim=-1)   # (batch,)
        max_coord = max_coord.clamp(min=0.0, max=self.box_max)

        level = torch.floor(max_coord / self.delta).long()  # 0..L
        level = level.clamp(min=0, max=self.L)
        return level

    def _exit_proba_from_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, dim)
        p_k = 1 / (L + 1 - k)
        """
        device = x.device
        level = self._level(x)                            # (batch,)
        denom = (self.L + 1 - level).to(torch.float32)    # (batch,)
        exit_proba = 1.0 / denom
        return exit_proba.to(device)

    def to_dist(self, x: torch.Tensor):
        """
        x: (batch, dim)
        return:
          exit_proba: (batch,)
          dist:       Beta 분포 (여기선 Beta(1,1), uniform on [0,1])
        """
        device = x.device
        batch_size = x.shape[0]

        # s0 처리: 배치 전체가 원점일 때
        if torch.all(x[0] == 0.0):
            assert torch.all(
                x == 0.0
            ), "If one of the states is s0, all of them must be s0 in this branch."

            # Never exit at s0
            exit_proba = torch.zeros(batch_size, device=device)

            # 원래 코드 스타일 유지: scalar Beta(1,1) (broadcast)
            dist = Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            )
            return exit_proba, dist

        # s0가 아닌 일반 상태들: 위치에 따라 exit_proba 부여
        exit_proba = self._exit_proba_from_state(x)

        # 각 state마다 independent Beta(1,1)
        dist = Beta(
            torch.ones(batch_size, device=device),
            torch.ones(batch_size, device=device),
        )
        return exit_proba, dist


class CirclePB(NeuralNet):
    def __init__(
        self,
        hidden_dim=64,
        n_hidden=2,
        torso=None,
        uniform=False,
        beta_min=0.1,
        beta_max=2.0,
    ):
        output_dim = 2  # Only alpha and beta for single Beta distribution
        super().__init__(
            dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim
        )
        if torso is not None:
            self.torso = torso
        self.uniform = uniform
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        # x is a batch of states, a tensor of shape (batch_size, dim) with dim == 2
        out = super().forward(x)
        log_alpha = out[:, 0]
        log_beta = out[:, 1]
        return (
            self.beta_max * torch.sigmoid(log_alpha) + self.beta_min,
            self.beta_max * torch.sigmoid(log_beta) + self.beta_min,
        )

    def to_dist(self, x):
        if self.uniform:
            return Beta(torch.ones(x.shape[0], device=x.device), torch.ones(x.shape[0], device=x.device))
        alpha, beta = self.forward(x)
        dist = Beta(alpha, beta)
        return dist