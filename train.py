import os
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

# Importamos nuestros filtros visuales y el secreto del puntaje
from wrappers import SkipFrame, GrayScaleResize, CustomReward


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback para guardar el mejor modelo durante el entrenamiento."""

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


def make_env(env_id, rank, seed=0):
    """Función de utilidad para crear entornos multiproceso optimizados."""

    def _init():
        # 1. Cargamos el juego correcto
        env = gym_super_mario_bros.make(env_id)

        # 2. Reducimos el espacio de acción a 7 botones
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # 3. Aplicamos los Wrappers de visión y recompensas
        env = SkipFrame(env, skip=4)
        env = GrayScaleResize(env, shape=(84, 84))
        env = CustomReward(env)

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    # ID exacto del nivel 1-1
    env_id = "SuperMarioBros-1-1-v0"
    num_cpu = 4  # Número de procesos en paralelo (ajusta según los núcleos de tu Linux)

    # Crear el entorno vectorizado y monitoreado
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), log_dir)

    # Configurar PPO con CnnPolicy (porque analiza imágenes)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Iniciar entrenamiento masivo
    model.learn(total_timesteps=5000000, callback=callback, tb_log_name="PPO-00003")

    model.save("mario_modelo_final")
    print("------------- Done Learning -------------")

    # Bloque de prueba final: ver a la IA jugar después de entrenar
    test_env = make_env(env_id, 0)()
    obs = test_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()
        if dones:
            obs = test_env.reset()