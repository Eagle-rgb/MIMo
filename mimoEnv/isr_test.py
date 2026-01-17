"""
Docstring for mimoEnv.isr_test

This file is used to test Initial State Randomization (ISR) in MIMo. To do this, it creates and saves images of
many resetted models so that the user can view them in their starting position. 
"""
import os
import gymnasium as gym
from render.utils import evaluation_video

import mimoEnv
import argparse
from mimoActuation.actuation import SpringDamperModel

from PIL import Image

def create_image(env, path, starting_position, img_num):
    print(f"Rendering image {img_num+1}")
    env.reset()
    frame = env.render()
    img = Image.fromarray(frame)
    img.save(os.path.join(path, f'irs_test_{starting_position}_{img_num+1}.png'))
    # take a random action to see if MIMo glitches.
    action = env.action_space.sample()
    env.step(action)
    img = Image.fromarray(env.render())
    img.save(os.path.join(path, f'irs_test_{starting_position}_{img_num+1}_x.png'))
    print(f"Achieved goal: {env.get_achieved_goal()}")
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_position', required=False,
                    choices=['supine', 'prone', 'alternating'],
                    default='prone',
                    help='Choose the starting position of MIMo in the roll_over environment. Put '
                            'either \'supine\', \'prone\' or \'alternating\'. Default: \'prone\'.')
    args = parser.parse_args()
    starting_position=args.starting_position

    env = gym.make("MIMoRollOver-v0", actuation_model=SpringDamperModel,
        starting_position=starting_position,
        width=480, # always 480 regardless whether we render actuations or not.
        height=480,
        render_mode='rgb_array',
        isr=True)
    
    images = []
    for i in range(10):
        images.append(create_image(env, '.', starting_position, i))

    evaluation_video(images, f"{starting_position}_vid.mp4", frame_rate=1, resolution=(480, 480))
