from mimoActuation.actuation import SpringDamperModel
import numpy as np
from gymnasium import spaces

class SpringDamperModel_PC1(SpringDamperModel):
    """ Class for the Spring-Damper actuation model.

    In this model, MIMo's muscles are represented by torque motors with linear and instantaneous control response, i.e.
    the abstract model directly matches the in-simulation definitions.
    The force-velocity and force-length relationships of real muscles is approximated using damping and spring
    components in the joint definitions of MIMo. The maximum torque of the motors is set to the maximum voluntary
    isometric torque along the corresponding axis, with a control input of 1 representing maximum torque.

    In addition to the attributes from the base actuation class, there are two extra attributes:

    Attributes:
        control_input (np.ndarray): Contains the current control input.
        max_torque (np.ndarray): The maximum motor torques.
    """
    def __init__(self, env, actuators, pca):
        super().__init__(env, actuators)
        self.pca = pca

        # 'self.control_input' is a zero vector when calling this. This makes it so that
        # when we get a 0 - activation of the PCA, we also get a 0 action. This makes the
        # model stable.
        self.pca.mean_ = self.control_input

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        The actuation space directly corresponds to the control range of the simulations motors. Unless modified, this
        will be [-1, 1] for all motors.

        Returns:
            gym.spaces.Space: The actuation space.
        """
        super().get_action_space()
        return spaces.Box(low=-100, high=100, dtype=np.float32)
    
    def action(self, action):
        """ Set the control inputs for the next step.

        Control values are clipped to the control range limits defined the MuJoCo XMLs and normalized to be even in
        both directions, i.e. an input of 0 corresponds to the center of the control range, rather than the default or
        neutral control position. The control ranges for the MIMo XMLs are set up to be symmetrical, such that an input
        of 0 corresponds to no motor torque.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        # 1. Erstelle einen leeren PC-Vektor (alle 30+ Komponenten auf 0)
        pca_space = np.zeros((1, self.pca.n_components_))
        
        # 2. Setze die erste Komponente auf den Output des Netzwerks
        pca_space[0, 0] = action[0]
        
        # 3. Transformiere zurück in den Raum der 44 Motoren
        full_action = self.pca.inverse_transform(pca_space)

        super().action(full_action.flatten())