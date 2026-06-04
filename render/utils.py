import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import mujoco
import cv2
import yaml
from skimage.transform import resize
import trimesh
import mimoEnv.utils as env_utils
from PIL import Image
import sys
# from babybench.build_xml import build

EPS=1e-6

def create_renderer(model):
    """ Creates mujoco render object. """
    renderer = mujoco.Renderer(model, height=240, width=240)
    return renderer

def create_top_down_camera(roll_over_starting_position='prone'):
    """ Creates and returns camera [mujoco.MjvCamera] object
    that looks top-down.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.distance = 1.1
    cam.elevation = -90

    # We must rotate 180° because else MIMo's head is at the bottom of
    # the screen for supine starting position
    cam.azimuth = 180 if roll_over_starting_position=='supine' else 0
    return cam

def get_surrounding_rect_center(data):
    """ Returns the center of the rectangle that is the projection
    of the surrounding cube of MIMo. Alternatively, you can think
    of this function drawing a top-down rectangle around MIMo and
    returning its center. Returns it as x, y tuple. """
    x_coords = data.xpos[:,0]
    y_coords = data.xpos[:,1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

def render_top_down(data, renderer, top_down_camera):
    """ Renders and returns the RGB array of MIMo top-down.
    Supply 'env.data' and the renderer created using
    'create_renderer' and the camera created using
    'create_top_down_camera'. """
    x, y = get_surrounding_rect_center(data)
    top_down_camera.lookat[0] = x
    top_down_camera.lookat[1] = y

    # cam.lookat = wrapped_env.unwrapped.data.body('mimo_location').xpos
    renderer.update_scene(data, camera=top_down_camera)
    pixels = renderer.render()
    return pixels

def render_top_down_and_save(env, starting_position, save_dir, name):
    """ Renders the current environment and saves it as '.pdf' file
    with filename 'name'.
    
    Creates a one-time use renderer and camera. Requires 'starting_position'
    to correctly orient the top-down camera. """
    if name[-4:] != ".pdf":
        name += ".pdf"

    renderer = create_renderer(env.model)
    cam = create_top_down_camera(starting_position)

    save_name = os.path.join(save_dir, name)
    frame = render_top_down(env.data, renderer, cam)
    Image.fromarray(frame).save(save_name)

def render(env, camera="corner"):
    img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name=camera)
    return img.astype(np.uint8)

def evaluation_img(env, up='side2', down='top'):
    img = np.zeros((480,720,3))
    img_corner = render(env, "corner")
    img[:,:480,:] = img_corner
    # Down-right rendering
    if down in ['top', 'side1', 'side2', 'closeup']:
        img_down = render(env, down)
        img[240:,480:,:] = img_down[::2,::2,:]
    elif down == 'binocular':
        img[240:,480:,:] =  view_binocular(env)
    elif down == 'actuations':
        img[240:,480:,:] = view_actuations(env)
    # Up-right rendering
    if up in ['top', 'side1', 'side2', 'closeup']:
        img_top = render(env, up)
        img[:240,480:,:] = img_top[::2,::2,:] 
    elif up == 'closeup':
        img_close = render(env, "closeup")
        img[:240,480:,:] = img_close[::2,::2,:]
    elif up == 'actuations':
        img[:240,480:,:] = view_actuations(env)
    elif up == 'binocular':
        img[:240,480:,:] = view_binocular(env)
    return img.astype(np.uint8)

def evaluation_video(images, save_name=None, frame_rate=60, resolution=((720,480))):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_name, fourcc, frame_rate, resolution)
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()

def view_binocular(env):
    img_left_eye = render(env, "eye_left")
    img_right_eye = render(env, "eye_right")
    stereo = np.zeros((240,240,3))
    stereo[:,:,0] = to_grayscale(img_left_eye[::2,::2,:])
    stereo[:,:,1] = to_grayscale(img_right_eye[::2,::2,:])
    stereo[:,:,2] = to_grayscale(img_right_eye[::2,::2,:])
    return stereo

def view_actuations(env, focus_body='hip', act_thresh=1e-6, abs_actuation=True):
    """ Plots actuation strenghts of each actuator (i.e. joint) of MIMo.

    Params:
        env (MIMoEnv): The mimo environment.
        focus_body (str): The body part to focus. Default: 'hip'.
        act_thresh (float): The absolute (!) threshold at which an actuator is drawn as gray
            (meaning 'not actuated').
        abs_actuation (bool): Whether to display absolute actuations (only red/gray) or
            include negative actuations. If false, negative actuations are drawn too and
            they are displayed in blue.
    """
    plt_data = np.array(env_utils.get_actuation_values(env.model, env.data))

    # Partition in gray/red points: Gray points where actuation is below threshold.
    if abs_actuation:
        indx_red = [abs(entry['actuation']) >= act_thresh for entry in plt_data]
        indx_gray = [abs(entry['actuation']) < act_thresh for entry in plt_data]
        indx_blue = []
    else:
        indx_red = [entry['actuation'] >= act_thresh for entry in plt_data]
        indx_gray = [abs(entry['actuation']) < act_thresh for entry in plt_data]
        indx_blue = [entry['actuation'] <= -act_thresh for entry in plt_data]

    plt_data_red = plt_data[indx_red]
    plt_data_gray = plt_data[indx_gray]
    plt_data_blue = plt_data[indx_blue]

    xs_red = np.array([entry['x'] for entry in plt_data_red])
    ys_red = np.array([entry['y'] for entry in plt_data_red])
    zs_red = np.array([entry['z'] for entry in plt_data_red])
    if abs_actuation:
        acts_red = np.array([abs(entry['actuation']) for entry in plt_data_red])
    else:
        acts_red = np.array([entry['actuation'] for entry in plt_data_red])

    xs_gray = np.array([entry['x'] for entry in plt_data_gray])
    ys_gray = np.array([entry['y'] for entry in plt_data_gray])
    zs_gray = np.array([entry['z'] for entry in plt_data_gray])
    acts_gray = np.array([entry['actuation'] for entry in plt_data_gray])

    xs_blue = np.array([entry['x'] for entry in plt_data_blue])
    ys_blue = np.array([entry['y'] for entry in plt_data_blue])
    zs_blue = np.array([entry['z'] for entry in plt_data_blue])
    acts_blue = np.array([entry['actuation'] for entry in plt_data_blue])

    # Subtract to focus on the hip.
    if focus_body:
        xs_red -= env.data.body(focus_body).xpos[0]
        ys_red -= env.data.body(focus_body).xpos[1]
        zs_red -= env.data.body(focus_body).xpos[2]
        xs_gray -= env.data.body(focus_body).xpos[0]
        ys_gray -= env.data.body(focus_body).xpos[1]
        zs_gray -= env.data.body(focus_body).xpos[2]
        xs_blue -= env.data.body(focus_body).xpos[0]
        ys_blue -= env.data.body(focus_body).xpos[1]
        zs_blue -= env.data.body(focus_body).xpos[2]

    if len(acts_red) > 0:
        opacities_red = acts_red
        red_colors = np.tile(np.array([1.0, 0, 0, 0]), (xs_red.shape[0], 1))
        red_colors[:, 3] = opacities_red

    if len(acts_blue) > 0:
        opacities_blue = acts_blue * (-1)
        blue_colors = np.tile(np.array([0, 0, 1.0, 0]), (xs_blue.shape[0], 1))
        blue_colors[:, 3] = opacities_blue

    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=0, roll=0)
    # Draw sensor points
    ax.scatter(xs_gray, ys_gray, zs_gray, color="k", s=5, depthshade=False, alpha=.15)

    if len(xs_red) > 0:
        ax.scatter(xs_red, ys_red, zs_red, color=red_colors, s=5, depthshade=False)

    if len(xs_blue) > 0:
        ax.scatter(xs_blue, ys_blue, zs_blue, color=blue_colors, s=5, depthshade=False)

    ax.set_xlim([-0.75, 0.75])     
    ax.set_ylim([-0.75, 0.75])     
    ax.set_zlim([-0.75, 0.75])     
    ax.set_axis_off()

    # Draw contact points
    # if contact_with is not None:
    #     if contact_with == 'hands':
    #         contact_checks = np.concatenate([env.right_hand_geoms, env.left_hand_geoms])

    #     contacts = env.data.contact
    #     for idx in range(len(contacts.geom1)):
    #         # Check if hands in contact
    #         if (contacts.geom1[idx] in contact_checks) and (contacts.geom2[idx] in env.mimo_geoms) \
    #         or (contacts.geom2[idx] in contact_checks) and (contacts.geom1[idx] in env.mimo_geoms):
    #             contact_position = contacts.pos[idx]
    #             ax.scatter(contact_position[0], contact_position[1], contact_position[2],
    #                        color="y", s=20, depthshade=True, alpha=0.8)
            
    fig.canvas.draw()
    plt.close()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = image_from_plot[160:160+240,195:195+240,:]
    #image_from_plot = (resize(image_from_plot, (240,240,3))*255).astype(np.uint8)
    return image_from_plot
    
def view_touches(env, focus_body='hip', contact_with=None):
    root_id = env_utils.get_body_id(env.model, body_name='mimo_location')
    points_no_contact = []
    points_contact = []
    contact_magnitudes = []
    # Go through all bodies and note their child bodies
    subtree = env_utils.get_child_bodies(env.model, root_id)
    for body_id in subtree:
        if (body_id in env.touch.sensor_positions) and (body_id in env.touch.sensor_outputs):
            sensor_points = env.touch.sensor_positions[body_id]
            force_vectors = env.touch.sensor_outputs[body_id]
            force_magnitude = np.linalg.norm(force_vectors, axis=-1, ord=2)
            no_touch_points = sensor_points[force_magnitude <= 1e-7]
            touch_points = sensor_points[force_magnitude > 1e-7]
            no_touch_points = env_utils.body_pos_to_world(env.data, position=no_touch_points, body_id=body_id)
            touch_points = env_utils.body_pos_to_world(env.data, position=touch_points, body_id=body_id)

            points_no_contact.append(no_touch_points)
            points_contact.append(touch_points)
            contact_magnitudes.append(force_magnitude[force_magnitude > 1e-7])

    points_gray = np.concatenate(points_no_contact)
    points_red = np.concatenate(points_contact)
    forces = np.concatenate(contact_magnitudes)
    if len(forces) > 0:
        size_min = 5
        size_max = 10
        sizes = forces / np.amax(forces) * (size_max - size_min) + size_min
        opacity_min = 0.4
        opacity_max = 0.5
        opacities = forces / np.amax(forces) * (opacity_max - opacity_min) + opacity_min
        # Opacities can't be set as an array, so must be set using color array
        red_colors = np.tile(np.array([1.0, 0, 0, 0]), (points_red.shape[0], 1))
        red_colors[:, 3] = opacities
    else:
        sizes = 5
        red_colors = [0.4,0,0]

    if focus_body:
        target_pos = env.data.body(focus_body).xpos
    else:
        target_pos = np.zeros((3,))

    # Subtract all by ball position to center on ball
    xs_gray = points_gray[:, 0] - target_pos[0]
    ys_gray = points_gray[:, 1] - target_pos[1]
    zs_gray = points_gray[:, 2] - target_pos[2]

    xs_red = points_red[:, 0] - target_pos[0]
    ys_red = points_red[:, 1] - target_pos[1]
    zs_red = points_red[:, 2] - target_pos[2]

    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=0, roll=0)
    # Draw sensor points
    ax.scatter(xs_gray, ys_gray, zs_gray, color="k", s=10, depthshade=False, alpha=.15)
    ax.scatter(xs_red, ys_red, zs_red, color=red_colors, s=sizes, depthshade=False)
    ax.set_xlim([-0.75, 0.75])     
    ax.set_ylim([-0.75, 0.75])     
    ax.set_zlim([-0.75, 0.75])     
    ax.set_axis_off()

    # Draw contact points
    if contact_with is not None:
        if contact_with == 'hands':
            contact_checks = np.concatenate([env.right_hand_geoms, env.left_hand_geoms])

        contacts = env.data.contact
        for idx in range(len(contacts.geom1)):
            # Check if hands in contact
            if (contacts.geom1[idx] in contact_checks) and (contacts.geom2[idx] in env.mimo_geoms) \
            or (contacts.geom2[idx] in contact_checks) and (contacts.geom1[idx] in env.mimo_geoms):
                contact_position = contacts.pos[idx]
                ax.scatter(contact_position[0], contact_position[1], contact_position[2],
                           color="y", s=20, depthshade=True, alpha=0.8)
            
    fig.canvas.draw()
    plt.close()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = image_from_plot[160:160+240,195:195+240,:]
    #image_from_plot = (resize(image_from_plot, (240,240,3))*255).astype(np.uint8)
    return image_from_plot

def to_grayscale(x):
    return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def make_save_dirs(save_dir):
    make_dir(save_dir)
    dirs = ['logs','videos']
    for dir_name in dirs:
        make_dir(f'{save_dir}/{dir_name}')

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)