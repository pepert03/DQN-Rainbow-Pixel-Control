At a high level you’re already halfway to Rainbow: you have switches for Double DQN, Dueling, and Prioritized Replay. To get a “Rainbow-style” agent with simple ablations, you need to:

- Add flags + code paths for:
  - Noisy Networks
  - n‑step returns
  - Distributional RL (C51-style)
- Make each component togglable (like your current `enable_double_dqn`, `enable_dueling_dqn`, `enable_prioritized_replay`) so you can flip them on/off in the YAML.

Below is a concise checklist of what to change and where.

***

## 1. Config: add Rainbow flags

In your `hyperparameters.yml` for Rainbow runs, add:

```yaml
rainbow_states_walker2d:
  env_id: Walker2d-v5
  obs: state
  replay_memory_size: 200000
  mini_batch_size: 64
  epsilon_init: 1.0
  epsilon_decay: 0.99999
  epsilon_min: 0.01
  network_sync_rate: 2000
  learning_rate: 0.0001
  discount_factor_g: 0.99

  enable_double_dqn: true
  enable_dueling_dqn: true
  enable_prioritized_replay: true

  # New Rainbow components
  enable_noisy_nets: true
  enable_distributional: true
  enable_n_step: true
  n_step: 3           # typical value: 3 or 5

  # Distributional parameters (C51-style)
  v_min: -100.0
  v_max: 100.0
  num_atoms: 51
```

And in `Agent.__init__` read them:

```python
self.enable_noisy_nets = config.get("enable_noisy_nets", False)
self.enable_distributional = config.get("enable_distributional", False)
self.enable_n_step = config.get("enable_n_step", False)
self.n_step = int(config.get("n_step", 1))

self.v_min = float(config.get("v_min", -100.0))
self.v_max = float(config.get("v_max", 100.0))
self.num_atoms = int(config.get("num_atoms", 51))
```

This keeps the same pattern as your existing switches and makes ablation easy (change booleans per experiment).

***

## 2. Noisy Nets: modify `DQN` (and optionally `Pixel_DQN`)

Add a small `NoisyLinear` module in `dqn.py`, then conditionally use it instead of `nn.Linear` in the last layers when `enable_noisy_nets` is true. That gives you Rainbow‑style parameterised exploration instead of \(\epsilon\)-greedy.

Minimal `NoisyLinear` (factorised noise):

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 / self.out_features**0.5)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
```

Then in `DQN.__init__`:

```python
def __init__(self, state_dim, action_dim, hidden_dim=512,
             enable_dueling_dqn=False, enable_noisy_nets=False,
             enable_distributional=False, num_atoms=51):
    super().__init__()
    self.enable_dueling_dqn = enable_dueling_dqn
    self.enable_noisy_nets = enable_noisy_nets
    self.enable_distributional = enable_distributional
    self.num_atoms = num_atoms

    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    Linear = NoisyLinear if enable_noisy_nets else nn.Linear

    if enable_dueling_dqn:
        self.fc_value = Linear(hidden_dim, 256)
        out_v = 1 * (num_atoms if enable_distributional else 1)
        self.value = Linear(256, out_v)

        self.fc_advantage = Linear(hidden_dim, 256)
        out_a = action_dim * (num_atoms if enable_distributional else 1)
        self.advantage = Linear(256, out_a)
    else:
        out_q = action_dim * (num_atoms if enable_distributional else 1)
        self.fc3 = Linear(hidden_dim, out_q)
```

In `forward`, if `enable_distributional` is false, keep current behaviour; if true, output logits per atom and reshape (see next section).

You can leave `Pixel_DQN` as-is at first (noisy nets are not mandatory for pixel experiments), or apply the same `Linear = NoisyLinear` trick to its last 2 fully connected layers.

Also: call `reset_noise()` on every training step if using noisy nets:

```python
if self.enable_noisy_nets:
    policy_dqn.apply(lambda m: isinstance(m, NoisyLinear) and m.reset_noise())
    target_dqn.apply(lambda m: isinstance(m, NoisyLinear) and m.reset_noise())
```

You can put that at the start of `optimize`.

***

## 3. Distributional RL (C51): change network output + loss

Rainbow’s distributional part (C51) does:

- Network outputs **logits** over `num_atoms` fixed supports \(z_i\) for each action.
- You compute a **projected Bellman target distribution** and train with cross‑entropy (KL).

### 3.1 Network output

Extend `DQN.forward`:

```python
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    if self.enable_dueling_dqn:
        v = F.relu(self.fc_value(x))
        V = self.value(v)

        a = F.relu(self.fc_advantage(x))
        A = self.advantage(a)

        if self.enable_distributional:
            # reshape: [B, 1*num_atoms] -> [B, 1, num_atoms]
            V = V.view(-1, 1, self.num_atoms)
            # [B, A*num_atoms] -> [B, A, num_atoms]
            A = A.view(-1, self.action_dim, self.num_atoms)
            Q = V + A - A.mean(dim=1, keepdim=True)
            return Q  # logits per atom
        else:
            Q = V + A - A.mean(dim=1, keepdim=True)
            return Q
    else:
        Q = self.fc3(x)
        if self.enable_distributional:
            Q = Q.view(-1, self.action_dim, self.num_atoms)
        return Q
```

For **action selection** in `Agent.run`:

```python
with torch.no_grad():
    q_logits = policy_dqn(state_tensor)
    if policy_dqn.enable_distributional:
        # expectation over atoms
        support = torch.linspace(self.v_min, self.v_max,
                                 policy_dqn.num_atoms, device=device)
        probs = q_logits.softmax(dim=2)  # [B, A, N_atoms]
        q_values = (probs * support).sum(dim=2)  # [B, A]
    else:
        q_values = q_logits
    action = q_values.squeeze(0).argmax().item()
```

### 3.2 Distributional loss in `optimize`

If `enable_distributional` is off, keep current MSE/weighted MSE.

If it’s on, replace:

- `target_q` computation with categorical projection.
- `current_q` with per‑action logits for chosen actions.
- `loss` with cross‑entropy.

Sketch (keep it simple; you can refine later):

```python
if policy_dqn.enable_distributional:
    batch_size = states.size(0)
    support = torch.linspace(self.v_min, self.v_max,
                             policy_dqn.num_atoms, device=device)
    delta_z = (self.v_max - self.v_min) / (policy_dqn.num_atoms - 1)

    # Next-state probs
    with torch.no_grad():
        next_logits = policy_dqn(new_states) if self.enable_double_dqn else target_dqn(new_states)
        next_probs = next_logits.softmax(dim=2)  # [B, A, N]
        # greedy actions using expectations
        next_q = (next_probs * support).sum(dim=2)  # [B, A]
        next_actions = next_q.argmax(dim=1, keepdim=True)  # [B,1]

        target_next_logits = target_dqn(new_states)
        target_next_probs = target_next_logits.softmax(dim=2)
        target_next_probs = target_next_probs.gather(
            1, next_actions.unsqueeze(-1).expand(-1, -1, policy_dqn.num_atoms)
        ).squeeze(1)  # [B, N]

        # project
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.discount_factor_g * support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros_like(target_next_probs)
        for i in range(policy_dqn.num_atoms):
            l_idx = l[:, i]
            u_idx = u[:, i]
            eq_mask = (u_idx == l_idx)
            m[torch.arange(batch_size), l_idx] += target_next_probs[:, i] * (u_idx - b[:, i] + eq_mask)
            m[torch.arange(batch_size), u_idx] += target_next_probs[:, i] * (b[:, i] - l_idx)

    # current logits for chosen actions
    logits = policy_dqn(states)  # [B, A, N]
    logits = logits.gather(
        1, actions.view(-1, 1, 1).expand(-1, 1, policy_dqn.num_atoms)
    ).squeeze(1)  # [B, N]

    log_probs = F.log_softmax(logits, dim=1)
    per_sample_loss = -(m * log_probs).sum(dim=1)  # [B]

    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        loss = (w * per_sample_loss).mean()
    else:
        loss = per_sample_loss.mean()
```

And skip the `current_q`, `td_errors`, MSE part when `enable_distributional` is true. Note: for PER with distributional RL, you can define `td_errors` as KL or the difference in expectations.

This is the heaviest change, but it’s isolated to `optimize` and `DQN.forward`.

***

## 4. n‑step returns: change buffer + target

Rainbow uses n‑step return to propagate reward faster.

### 4.1 Add n‑step buffer

Simplest: wrap your existing replay buffer with a small n‑step queue in `Agent.run`. When `enable_n_step` is true, you don’t push single‑step transitions directly; you accumulate them:

- Maintain a deque of the last `n_step` transitions: `(s_t, a_t, r_t, ...)`.
- When it’s full or episode ends, build `R_t^{(n)}`:

\[
R_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1}
\]

- Push `(s_t, a_t, R_t^{(n)}, s_{t+n}, done_{t+n-1})` into replay.

There are many ways to implement this; the important bit for you is that in `optimize` you then use `discount_factor_g**n_step` instead of `gamma` for the bootstrap part:

```python
gamma_n = self.discount_factor_g ** self.n_step

# target_q when NOT distributional:
target_q = rewards + (1 - dones) * gamma_n * next_q
```

You can start without n‑step to keep things simpler, then add it once distributional and noisy nets are working.

***

## 5. Ablation: using the flags

Once the above wiring is in place, ablations are just different YAML entries:

- Full Rainbow:
  - `enable_double_dqn: true`
  - `enable_dueling_dqn: true`
  - `enable_prioritized_replay: true`
  - `enable_noisy_nets: true`
  - `enable_distributional: true`
  - `enable_n_step: true`

- Rainbow – PER:
  - Set `enable_prioritized_replay: false`, keep everything else.

- Rainbow – Noisy:
  - Set `enable_noisy_nets: false` and revert to \(\epsilon\)-greedy for exploration.

- Rainbow – Distributional:
  - `enable_distributional: false` (fall back to scalar Q + MSE).

Because the logic is modular in `Agent.optimize` and `DQN.forward`, you won’t need to touch the main training loop.