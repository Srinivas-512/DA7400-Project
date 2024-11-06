import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

TensorBatch = List[torch.Tensor]

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._traj_labels = torch.zeros((buffer_size, 1), dtype=torch.long, device=device)
        self._traj_end_indices = torch.zeros((buffer_size, 1), dtype=torch.long, device=device)
        self._traj_start_indices = torch.zeros((buffer_size, 1), dtype=torch.long, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        data = {key: value[:1000] for key, value in data.items()}
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        # self._traj_labels[:n_transitions] = torch.cumsum(self._dones[:n_transitions], dim=0)

        curr_end = n_transitions - 1
        curr_start = 0

        for i in tqdm(range(n_transitions-1, -1, -1)):
            j = n_transitions -1 -i
            self._traj_end_indices[i] = curr_end
            self._traj_start_indices[j] = curr_start
            if self._dones[i]:
                curr_end = i
            if self._dones[j]:
                curr_start = j+1

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        start_indices = self._traj_start_indices[indices]
        end_indices = self._traj_end_indices[indices]
        return [states, actions, rewards, next_states, dones, indices, start_indices, end_indices]