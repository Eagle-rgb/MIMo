from mimoActuation.actuation import SpringDamperModel
import numpy as np

class SpringDamperModel_Stationary_Limbs(SpringDamperModel):
    """ This class extends SpringDamperModel in the following way:

    It allows setting bool variables 'frozen_LA' and 'frozen_LL' corresponding
    to left arm and left leg. These bool variables are by default False, but setting
    them freezes all actuators in their group. This file is used to test whether MIMo
    can perform Supine -> Prone rollover with some limbs frozen ('stationary' as by
    Kobayashi 2016). It was difficult to classify stationary / non-stationary limbs, so
    we rather just test if MIMo can do a rollover with some limbs stationary.

    In addition to the attributes from the base actuation class, there are two extra attributes:

    Attributes:
        control_input (np.ndarray): Contains the current control input.
        max_torque (np.ndarray): The maximum motor torques.
    """
    # Define groups of actuators to freeze.
    ACT_LEFT_ARM = ["act:left_shoulder_horizontal",
                    "act:left_shoulder_abduction",
                    "act:left_shoulder_internal",
                    "act:left_elbow",
                    "act:left_wrist_rotation",
                    "act:left_wrist_flexion",
                    "act:left_wrist_ulnar",
                    "act:left_fingers"]
    
    ACT_LEFT_LEG = ['act:left_hip_flex',
                    'act:left_hip_abduction',
                    'act:left_hip_rotation',
                    'act:left_knee',
                    'act:left_foot_flexion',
                    'act:left_foot_inversion',
                    'act:left_foot_rotation',
                    'act:left_toes']

    def __init__(self, env, actuators, freeze_leg, freeze_arm):
        super().__init__(env, actuators)

        # find actuator indexes for each group. These contain - in order - the indexes
        # of the actuators found in the static lists above. The indexes are in the
        # 'actuators' list. We can then simply set the control to 0 at that index.
        self.act_left_arm_idx = []
        self.act_left_leg_idx = []
        self.freeze_arm = freeze_arm
        self.freeze_leg = freeze_leg

        for act_name in self.ACT_LEFT_ARM:
            act_found=False
            for i in range(len(actuators)):
                act_idx = actuators[i]
                tmp_act_name = env.model.actuator(act_idx).name
                if tmp_act_name == act_name:
                    self.act_left_arm_idx.append(i)
                    act_found=True
                    break

            if not act_found:
                raise RuntimeError(f"{act_name} not found")
            
        for act_name in self.ACT_LEFT_LEG:
            act_found=False
            for i in range(len(actuators)):
                act_idx = actuators[i]
                tmp_act_name = env.model.actuator(act_idx).name
                if tmp_act_name == act_name:
                    self.act_left_leg_idx.append(i)
                    act_found=True
                    break

            if not act_found:
                raise RuntimeError(f"{act_name} not found")
    
    def action(self, action):
        if self.freeze_arm:
            action[self.act_left_arm_idx] = np.zeros(len(self.act_left_arm_idx))

        if self.freeze_leg:
            action[self.act_left_leg_idx] = np.zeros(len(self.act_left_leg_idx))

        super().action(action)