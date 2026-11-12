import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO

# Importamos exactamente los mismos filtros visuales que usamos en train.py
from wrappers import SkipFrame, GrayScaleResize

# Cargamos el mejor modelo que se guardó durante el entrenamiento
# Nota: Stable Baselines lee los archivos .zip automáticamente
model_path = "tmp/best_model"
model = PPO.load(model_path)


def main():
    steps = 0
    # 1. Cargamos el entorno nativo de Mario
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 2. APLICAMOS LA MISMA VISIÓN DE IA
    # Esto es crítico para que las neuronas del modelo coincidan con la imagen
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env, shape=(84, 84))

    obs = env.reset()
    info = {}  # Inicializamos info vacía para evitar advertencias de PyCharm

    print("¡Despertando a la IA! Observa cómo juega...")

    # Usamos un bucle infinito para que siga jugando partida tras partida
    while True:
        # El modelo predice la mejor acción basándose en lo que "ve" (obs)
        action, _state = model.predict(obs)

        # Ejecutamos la acción en el emulador
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            print(f"Partida terminada. Score final: {info.get('score', 0)}. Reiniciando...")
            obs = env.reset()

        steps += 1
        if steps % 500 == 0:
            print(f"Total Steps: {steps} | Score actual: {info.get('score', 0)}")


if __name__ == "__main__":
    main()