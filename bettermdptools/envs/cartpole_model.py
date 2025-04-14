"""
generative AI experiment - discretized cartpole transition and reward (P) matrix with adaptive angle binning
created with chatGPT

Example usage:
dpole = DiscretizedCartPole(10, 10, 10, .1, .5)  # Example bin sizes for each variable and adaptive angle binning center/outer resolution

"""

import numpy as np
import warnings


class DiscretizedCartPole:
    def __init__(
        self,
        position_bins,
        velocity_bins,
        angular_velocity_bins,
        angular_center_resolution,
        angular_outer_resolution,
        num_angular_center_bins=10,
        timestep=0.02
    ):
        """
        Initializes the DiscretizedCartPole model.
        
        The range for each variable is fixed.
        - position: (-2.4, 2.4)
        - velocity: (-3, 3)
        - angle: (-12 * pi / 180, 12 * pi / 180) (-12 degrees to 12 degrees, expressed in radians)
        - angular_velocity: (-1.5, 1.5) (angular velocity in radians per second)

        Parameters:
        - position_bins (int): Number of discrete bins for the cart's position.
        - velocity_bins (int): Number of discrete bins for the cart's velocity.
        - angular_velocity_bins (int): Number of discrete bins for the pole's angular velocity.
        - angular_center_resolution (float): The resolution of angle bins The region of interest around zero for higher resolution.
        - angular_outer_resolution (float): The resolution of angle bins away from the center.
        - num_angular_center_bins (int): Number of bins around the center for higher resolution.
        
        Attributes:
        - state_space (int): Total number of discrete states in the environment.
        - P (dict): Transition probability matrix where P[state][action] is a list of tuples (probability, next_state,
        reward, done).
        - transform_obs (lambda): Function to transform continuous observations into a discrete state index.
        """
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.action_space = 2  # Left or Right

        # Define the range for each variable
        self.position_range = (-2.4, 2.4)
        self.velocity_range = (-3, 3)
        self.angle_range = (-12 * np.pi / 180, 12 * np.pi / 180)
        self.angular_velocity_range = (-1.5, 1.5)
        self.angular_center_resolution = angular_center_resolution
        self.angular_outer_resolution = angular_outer_resolution
        self.num_angular_center_bins = num_angular_center_bins
        
        # Cart physics. Should match the original CartPole environment.
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = timestep  # seconds between state updates

        # Use adaptive binning for the pole angle
        self.angle_bins = self.adaptive_angle_bins(
            self.angle_range,
            self.angular_center_resolution,
            self.angular_outer_resolution,
            num_angular_center_bins=self.num_angular_center_bins,
        )  # Adjust these values as needed

        self.state_space = np.prod(
            [
                self.position_bins,
                self.velocity_bins,
                len(self.angle_bins),
                self.angular_velocity_bins,
            ]
        )
        self.P = {
            state: {action: [] for action in range(self.action_space)}
            for state in range(self.state_space)
        }
        self.setup_transition_probabilities()
        self.n_states = (
            len(self.angle_bins)
            * self.velocity_bins
            * self.position_bins
            * self.angular_velocity_bins
        )
        """
        Explanation of transform_obs lambda: 
        This lambda function will take cartpole observations, determine which bins they fall into, 
        and then convert bin coordinates into a single index.  This makes it possible 
        to use traditional reinforcement learning and planning algorithms, designed for discrete spaces, with continuous 
        state space environments. 
        
        Parameters:
        - obs (list): A list of continuous observations [position, velocity, angle, angular_velocity].

        Returns:
        - int: A single integer representing the discretized state index.
        """
        self.transform_obs = lambda obs: (
            np.ravel_multi_index(
                (
                    np.clip(
                        np.digitize(
                            obs[0],
                            np.linspace(*self.position_range, self.position_bins),
                        )
                        - 1,
                        0,
                        self.position_bins - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[1],
                            np.linspace(*self.velocity_range, self.velocity_bins),
                        )
                        - 1,
                        0,
                        self.velocity_bins - 1,
                    ),
                    np.clip(
                        np.digitize(obs[2], self.angle_bins) - 1,
                        0,
                        len(self.angle_bins) - 1,
                    ),
                    # Use adaptive angle bins
                    np.clip(
                        np.digitize(
                            obs[3],
                            np.linspace(
                                *self.angular_velocity_range, self.angular_velocity_bins
                            ),
                        )
                        - 1,
                        0,
                        self.angular_velocity_bins - 1,
                    ),
                ),
                (
                    self.position_bins,
                    self.velocity_bins,
                    len(self.angle_bins),
                    self.angular_velocity_bins,
                ),
            )
        )

    def adaptive_angle_bins(self, angle_range, center_resolution, outer_resolution, num_angular_center_bins=10):
        """
        Generates adaptive bins for the pole's angle to allow for finer resolution near the center and coarser
        resolution farther away.

        Parameters:
        - angle_range (tuple): The minimum and maximum angles in radians.
        - center_resolution (float): Region of interest around zero angle for higher resolution.
        - outer_resolution (float): Bin width away from zero for lower resolution.

        Returns:
        - np.array: An array of bin edges with adaptive spacing.
        """
        min_angle, max_angle = angle_range
        # Generate finer bins around zero
        center_bin_size = center_resolution / num_angular_center_bins
        if center_bin_size >= outer_resolution:
            warnings.warn(
                f"Center bin size will be {center_bin_size} vs outer bin size of {outer_resolution}."
            )
        center_bins = np.arange(
            -center_resolution, center_resolution + 1e-6, center_bin_size
        )
        # Generate sparser bins outside the center region by dividing the remaining range
        # into bins of size `outer_resolution`
        left_distance = np.abs(np.abs(min_angle) - np.abs(center_resolution))
        left_bins = np.linspace(
            min_angle,
            -center_resolution,
            num=int(left_distance / outer_resolution) + 1,
            endpoint=True,
        )

        right_distance = np.abs(np.abs(max_angle) - np.abs(center_resolution))
        right_bins = np.linspace(
            center_resolution,
            max_angle,
            num=int(right_distance / outer_resolution) + 1,
            endpoint=True,
        )
        return np.unique(np.concatenate([left_bins, center_bins, right_bins]))

    def setup_transition_probabilities(self):
        """
        Sets up the transition probabilities for the environment. This method iterates through all possible
        states and actions, simulates the next state, and records the transition probability
        (deterministic in this setup), reward, and termination status.
        """
        for state in range(self.state_space):
            position, velocity, angle, angular_velocity = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(
                    position, velocity, angle, angular_velocity, action
                )
                self.P[state][action].append((1, next_state, reward, done))

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: A list of indices representing the state in terms of position, velocity, angle, and angular velocity bins.
        """
        totals = [
            self.position_bins,
            self.velocity_bins,
            len(self.angle_bins),
            self.angular_velocity_bins,
        ]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components
    
    def convert_state_indices_to_continuous(self, state_indices):
        """
        Converts state indices back to continuous values for position, velocity, angle, and angular velocity.

        Parameters:
        - state_indices (tuple): A tuple of indices representing the state.

        Returns:
        - list: A list of continuous values for position, velocity, angle, and angular velocity.
        """
        position_idx, velocity_idx, angle_idx, angular_velocity_idx = state_indices
        position = np.linspace(*self.position_range, self.position_bins)[position_idx]
        velocity = np.linspace(*self.velocity_range, self.velocity_bins)[velocity_idx]
        angle = self.angle_bins[angle_idx]
        angular_velocity = np.linspace(
            *self.angular_velocity_range, self.angular_velocity_bins
        )[angular_velocity_idx]

        return position, velocity, angle, angular_velocity

    def compute_next_state(
        self, position_idx, velocity_idx, angle_idx, angular_velocity_idx, action
    ):
        """
        Computes the next state based on the current state indices and the action taken. Applies simplified physics calculations to determine the next state.

        Parameters:
        - position_idx (int): Current index of the cart's position.
        - velocity_idx (int): Current index of the cart's velocity.
        - angle_idx (int): Current index of the pole's angle.
        - angular_velocity_idx (int): Current index of the pole's angular velocity.
        - action (int): Action taken (0 for left, 1 for right).

        Returns:
        - tuple: Contains the next state index, the reward, and the done flag indicating if the episode has ended.
        """
        position, velocity, angle, angular_velocity = self.convert_state_indices_to_continuous(
            (position_idx, velocity_idx, angle_idx, angular_velocity_idx)
        )
        
        # position = np.linspace(*self.position_range, self.position_bins)[position_idx]
        # velocity = np.linspace(*self.velocity_range, self.velocity_bins)[velocity_idx]
        # angle = self.angle_bins[angle_idx]
        # angular_velocity = np.linspace(
        #     *self.angular_velocity_range, self.angular_velocity_bins
        # )[angular_velocity_idx]

        # ORIGINAL
        # Simulate physics here (simplified)
        # force = 10 if action == 1 else -10
        # new_velocity = velocity + (force + np.cos(angle) * -10.0) * 0.02
        # new_position = position + new_velocity * 0.02
        # new_angular_velocity = angular_velocity + (-3.0 * np.sin(angle)) * 0.02
        # new_angle = angle + new_angular_velocity * 0.02
        
        # From Edward's code https://edstem.org/us/courses/71185/discussion/6518290
        # thresh = 0.1
        # force = 10 if action == 1 else -10
        # new_velocity = velocity + (force + np.sin(angle) * -10.0) * thresh
        # new_position = position + new_velocity * thresh
        # new_angular_velocity = angular_velocity + (3.0 * np.sin(angle) - force) * thresh
        # new_angle = angle + new_angular_velocity * thresh
        
        # Niels' code (based on the original CartPole physics)
        # Derived from Lagrange equations of motion for the cart-pole system
        force = self.force_mag if action == 1 else -self.force_mag
        
        # angular_acceleration = self.total_mass * self.gravity * sin_angle - cos_angle * (
        #     force + self.polemass_length * angular_velocity**2 * sin_angle
        # ) / (self.length * (self.total_mass - self.masspole * cos_angle**2))
        # acceleration = (force + self.polemass_length * (angular_velocity**2 * sin_angle - angular_acceleration * cos_angle)) / self.total_mass
        
        # taken from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
        sintheta = np.sin(angle)
        costheta = np.cos(angle)
        temp = (
            force + self.polemass_length * np.square(angular_velocity) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        new_position = position + velocity * self.tau
        new_velocity = velocity + xacc * self.tau 
        new_angle = angle + angular_velocity * self.tau
        new_angular_velocity = angular_velocity + thetaacc * self.tau

        new_position_idx = np.clip(
            np.digitize(
                new_position, np.linspace(*self.position_range, self.position_bins)
            )
            - 1,
            0,
            self.position_bins - 1,
        )
        new_velocity_idx = np.clip(
            np.digitize(
                new_velocity, np.linspace(*self.velocity_range, self.velocity_bins)
            )
            - 1,
            0,
            self.velocity_bins - 1,
        )
        new_angle_idx = np.clip(
            np.digitize(new_angle, self.angle_bins) - 1, 0, len(self.angle_bins) - 1
        )
        new_angular_velocity_idx = np.clip(
            np.digitize(
                new_angular_velocity,
                np.linspace(*self.angular_velocity_range, self.angular_velocity_bins),
            )
            - 1,
            0,
            self.angular_velocity_bins - 1,
        )

        new_state_idx = np.ravel_multi_index(
            (
                new_position_idx,
                new_velocity_idx,
                new_angle_idx,
                new_angular_velocity_idx,
            ),
            (
                self.position_bins,
                self.velocity_bins,
                len(self.angle_bins),
                self.angular_velocity_bins,
            ),
        )

        reward = 1 if np.abs(new_angle) < 12 * np.pi / 180 else -1
        done = (
            True
            if np.abs(new_angle) >= 12 * np.pi / 180 or np.abs(new_position) > 2.4
            else False
        )

        return new_state_idx, reward, done
