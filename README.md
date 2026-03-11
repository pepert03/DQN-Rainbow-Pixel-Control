
# DQN-Rainbow-Pixel-Control

Entrenamiento de agentes **DQN** y **Rainbow DQN** sobre tareas MuJoCo usando Gymnasium, con dos modos de observación:

- **state**: vectores de estado de baja dimensionalidad.
- **pixel**: observaciones visuales RGB (redimensionadas + frame-stacking).

El sistema detecta automáticamente si usar vanilla DQN o Rainbow según los flags de la configuración. El entrenamiento se paraleliza con **entornos vectorizados** (`SyncVectorEnv`) y las métricas se registran en **TensorBoard**.

---

## Fundamento teórico

### DQN

La idea central de **Deep Q-Learning** es aproximar la función de valor-acción óptima $Q^*(s, a)$ con una red neuronal $Q(s, a; \theta)$.

**Ecuación de optimalidad de Bellman:**

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

**TD target con red objetivo** ($\theta^-$):

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**Pérdida:**

$$L(\theta) = \mathbb{E}\left[(y - Q(s, a; \theta))^2\right]$$

Mecanismos de estabilización:

- **Experience Replay**: almacena transiciones $(s, a, r, s', done)$ y muestrea mini-batches.
- **Target Network**: copia periódica de pesos para estabilizar el target.

### Rainbow DQN

Rainbow combina seis extensiones sobre DQN vanilla:

| Extensión | Flag en config | Descripción |
|---|---|---|
| **Double DQN** | `enable_double_dqn` | Desacopla selección y evaluación de acción para reducir sobreestimación |
| **Dueling DQN** | `enable_dueling_dqn` | Separa el valor de estado $V(s)$ y la ventaja $A(s,a)$ |
| **Prioritized Replay** | `enable_prioritized_replay` | Muestrea transiciones proporcionalmente a su error TD |
| **Noisy Nets** | `enable_noisy_nets` | Reemplaza $\epsilon$-greedy por ruido paramétrico en las capas lineales |
| **Distributional (C51)** | `enable_distributional` | Modela la distribución completa del retorno en vez de solo la esperanza |
| **N-step Returns** | `enable_n_step` | Usa retornos multi-paso para propagar recompensas más rápido |

Si **cualquier** flag Rainbow está activo, `main.py` enruta automáticamente al trainer Rainbow; si todos están desactivados, usa vanilla DQN.

### Modo pixel: por qué Conv2D empieza con 12 canales

En modo pixel la observación tiene forma `[stack, H, W, C]`. Con `stack_size=4` y RGB (`C=3`), los canales de entrada son $4 \times 3 = 12$, de ahí `Conv2d(12, ...)`.

---

## Estructura del repositorio

```
.
├── main.py                        # Punto de entrada CLI (train / evaluate)
├── pyproject.toml                 # Dependencias y metadata del proyecto
├── configs/
│   └── hyperparameters.yml        # Definición de runs (presets)
├── runs/                          # Salidas generadas por run
│   └── <nombre_run>/
│       ├── config.yml             # Copia congelada de la configuración
│       ├── best_model.pt          # Mejor modelo (por reward episódico)
│       ├── checkpoint.pt          # Checkpoint periódico (reanudable)
│       ├── training.log           # Log de texto
│       ├── graph.png              # Gráfica de reward + epsilon
│       └── tensorboard/           # Logs de TensorBoard
├── src/
│   ├── __init__.py
│   ├── config.py                  # Carga de config + paths + device
│   ├── networks.py                # DQN (MLP), Pixel_DQN (CNN), NoisyLinear
│   ├── buffer.py                  # Experience Replay (buffer circular)
│   ├── wrappers.py                # Wrappers de entorno + factories
│   ├── utils.py                   # Checkpoint, save model, gráficas
│   ├── dqn/
│   │   ├── __init__.py
│   │   └── train.py               # Entrenamiento y evaluación DQN vanilla
│   └── rainbow/
│       ├── __init__.py
│       ├── buffer.py              # Prioritized Experience Replay
│       └── train.py               # Entrenamiento y evaluación Rainbow
├── notebooks/                     # Notebooks de exploración / debug
└── Report/                        # Informe LaTeX
```

### ¿Qué se guarda en `runs/<nombre_run>/`?

| Archivo | Descripción |
|---|---|
| `config.yml` | Copia de la configuración usada (se congela en la primera ejecución) |
| `best_model.pt` | Pesos del modelo con el mayor reward episódico |
| `checkpoint.pt` | Estado completo para reanudar (modelo, optimizador, epsilon, historial, loss) |
| `training.log` | Log de texto con timestamps |
| `graph.png` | Gráfica de recompensa media y epsilon |
| `tensorboard/` | Eventos de TensorBoard (reward, loss, epsilon, buffer size, ep/s) |

---

## Instalación

### Requisitos

- Python >= 3.10
- CUDA recomendado (se usa automáticamente si está disponible)

### Con `uv`

```bash
git clone https://github.com/pepert03/DQN-Rainbow-Pixel-Control
cd DQN-Rainbow-Pixel-Control
uv sync
```

---

## Uso

### 1) Definir una run en `configs/hyperparameters.yml`

Cada clave de nivel superior es el nombre de una run. Ejemplo mínimo:

```yaml
mi_experimento:
  env_id: Walker2d-v5
  obs: state                        # state | pixel
  replay_memory_size: 200000
  mini_batch_size: 64
  epsilon_init: 1.0
  epsilon_decay: 0.99999
  epsilon_min: 0.01
  network_sync_rate: 2000
  learning_rate: 0.0001
  discount_factor_g: 0.99
  num_envs: 30                      # Entornos paralelos (SyncVectorEnv)

  # Flags Rainbow (poner todos a false para DQN vanilla)
  enable_double_dqn: false
  enable_dueling_dqn: false
  enable_prioritized_replay: false
  enable_noisy_nets: false
  enable_distributional: false
  enable_n_step: false
```

> **Nota:** la primera vez que se ejecuta una run, la configuración se copia a `runs/<nombre_run>/config.yml` y queda congelada. Las ejecuciones siguientes leen desde `runs/` (no desde `configs/`). Para cambiar hiperparámetros hay que borrar `runs/<nombre_run>/config.yml` (o toda la carpeta).

### 2) Entrenar

```bash
uv run python main.py mi_experimento --train
```

La consola muestra en tiempo real:

```
Episode 1500 | 420.3 ep/s | Reward: 85.2 | Mean(100): 62.4 | Loss: 0.0342 | Epsilon: 0.8521
```

El entrenamiento se puede parar con `Ctrl+C` y reanudar ejecutando el mismo comando (se carga el checkpoint automáticamente).

### 3) Monitorizar con TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Métricas disponibles:

- `episode/reward`, `episode/mean_reward_100`, `episode/best_reward`
- `train/loss`, `train/epsilon`, `train/buffer_size`
- `perf/episodes_per_second`

### 4) Evaluar (con renderizado)

```bash
uv run python main.py mi_experimento
```

Carga `runs/<nombre_run>/best_model.pt` y renderiza el agente en el entorno.

- En modo **pixel**: `render_mode="rgb_array"` + `HumanRendering`.
- En modo **state**: `render_mode="human"`.

---

## Configuraciones incluidas

| Nombre | Entorno | Obs | Rainbow | `num_envs` |
|---|---|---|---|---|
| `states_walker2d` | Walker2d-v5 | state | No | 30 |
| `walker2d` | Walker2d-v5 | pixel | No | 12 |
| `rainbow_states_walker2d` | Walker2d-v5 | state | Sí (todos) | 12 |
| `rainbow_pixel_walker2d` | Walker2d-v5 | pixel | Sí (todos) | 12 |
| `cartpole` | CartPole-v1 | pixel | No | 12 |
| `rainbow_cartpole` | CartPole-v1 | pixel | Sí (todos) | 12 |
| `hopper2d` | Hopper-v5 | pixel | No | 12 |