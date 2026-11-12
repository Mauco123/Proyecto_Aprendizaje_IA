import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# 1. Creas el entorno
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 2. ¡LÍNEA FALTANTE! Aplicas la restricción de controles al entorno
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 3. Inicias el nivel
obs = env.reset()
print("Formato de la imagen (Alto, Ancho, Colores):", obs.shape)
done = False # Cuando termina la ejecución (muere o gana)

while not done:
    # Va a dar una acción random que el agente puede hacer (moverse, etc) de las 7 posibles
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()

env.close()