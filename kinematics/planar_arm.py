import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import create_jacobian, create_dh_matrix


class PlanarArm:
    # joint limits
    l_upper_arm_limit, u_upper_arm_limit = np.radians((-5., 175.))  # in degrees [°]
    l_forearm_limit, u_forearm_limit = np.radians((-5., 175.))  # in degrees [°]

    # DH parameter
    scale = 1.0
    shoulder_length = scale * 50.0  # in [mm]
    upper_arm_length = scale * 220.0  # in [mm]
    forearm_length = scale * 160.0  # in [mm]

    # visualisation parameters
    x_limits = (-450, 450)
    y_limits = (-50, 400)

    def __init__(self,
                 init_thetas: np.ndarray | tuple | list,
                 arm: str = 'right',
                 radians: bool = False):

        """Constructor: initialize current joint angles, positions and trajectories"""
        if isinstance(init_thetas, tuple | list):
            init_thetas = np.array(init_thetas)

        self.arm = arm
        self.angles = self.check_values(init_thetas, radians)
        self.trajectory = [init_thetas]
        self.end_effector = self.update_end_effector()

    def update_end_effector(self):
        self.end_effector = PlanarArm.forward_kinematics(arm=self.arm,
                                                         thetas=self.angles,
                                                         radians=True)[:, -1]
        return self.end_effector

    @staticmethod
    def check_values(angles: np.ndarray, radians: bool):
        assert angles.size == 2, "Arm must contain two angles: angle shoulder, angle elbow"

        if not radians:
            angles = np.radians(angles)

        if angles[0] <= PlanarArm.l_upper_arm_limit or angles[0] >= PlanarArm.u_upper_arm_limit:
            raise AssertionError('Check joint limits for upper arm')
        elif angles[1] <= PlanarArm.l_forearm_limit or angles[1] >= PlanarArm.u_forearm_limit:
            raise AssertionError('Check joint limits for forearm')

        return angles

    @staticmethod
    def clip_values(angles: np.ndarray, radians: bool):
        assert angles.size == 2, "Arm must contain two angles: angle shoulder, angle elbow"

        if not radians:
            angles = np.radians(angles)

        angles[0] = np.clip(angles[0], a_min=PlanarArm.l_upper_arm_limit, a_max=PlanarArm.u_upper_arm_limit)
        angles[1] = np.clip(angles[1], a_min=PlanarArm.l_forearm_limit, a_max=PlanarArm.u_forearm_limit)

        return angles if radians else np.degrees(angles)

    @staticmethod
    def __circular_wrap(x: float, x_min: int | float, x_max: int | float):
        # Calculate the range of the interval
        interval_range = x_max - x_min

        # Calculate the wrapped value of x
        wrapped_x = x_min + ((x - x_min) % interval_range)

        return wrapped_x

    @staticmethod
    def circ_values(thetas: np.ndarray, radians: bool = True):
        """
        This wrapper function is intended to prevent phase jumps in the inverse kinematics due to large errors in the
        gradient calculation. This means that joint angles are only possible within the given limits.

        :param thetas:
        :param radians:
        :return:
        """
        if not radians:
            theta1, theta2 = np.radians(thetas)
        else:
            theta1, theta2 = thetas

        theta1 = PlanarArm.__circular_wrap(x=theta1,
                                           x_min=PlanarArm.l_upper_arm_limit,
                                           x_max=PlanarArm.u_upper_arm_limit)

        theta2 = PlanarArm.__circular_wrap(x=theta2,
                                           x_min=PlanarArm.l_forearm_limit,
                                           x_max=PlanarArm.u_forearm_limit)

        return np.array((theta1, theta2))

    @staticmethod
    def forward_kinematics(arm: str, thetas: np.ndarray, radians: bool = False, check_limits: bool = True):

        if check_limits:
            theta1, theta2 = PlanarArm.check_values(thetas, radians)
        else:
            theta1, theta2 = thetas

        if arm == 'right':
            const = 1
        elif arm == 'left':
            const = - 1
            theta1 = np.pi - theta1
            theta2 = - theta2
        else:
            raise ValueError('Please specify if the arm is right or left!')

        A0 = create_dh_matrix(a=const * PlanarArm.shoulder_length, d=0,
                              alpha=0, theta=0)

        A1 = create_dh_matrix(a=PlanarArm.upper_arm_length, d=0,
                              alpha=0, theta=theta1)

        A2 = create_dh_matrix(a=PlanarArm.forearm_length, d=0,
                              alpha=0, theta=theta2)

        # Shoulder -> elbow
        A01 = A0 @ A1
        # Elbow -> hand
        A12 = A01 @ A2

        return np.column_stack(([0, 0], A0[:2, 3], A01[:2, 3], A12[:2, 3]))

    @staticmethod
    def inverse_kinematics(arm: str,
                           end_effector: np.ndarray,
                           starting_angles: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iterations: int = 5000,
                           abort_criteria: float = 1,  # in [mm]
                           radians: bool = False):

        if not radians:
            starting_angles = np.radians(starting_angles)

        thetas = starting_angles.copy()
        for i in range(max_iterations):
            # Compute the forward kinematics for the current joint angles
            current_position = PlanarArm.forward_kinematics(arm=arm,
                                                            thetas=thetas,
                                                            radians=True)[:, -1]

            # Calculate the error between the current end effector position and the desired end point
            error = end_effector - current_position

            # abort when error is smaller than the breaking condition
            if np.linalg.norm(error) < abort_criteria:
                break

            # Calculate the Jacobian matrix for the current joint angles
            J = create_jacobian(thetas=thetas, arm=arm,
                                a_sh=PlanarArm.upper_arm_length,
                                a_el=PlanarArm.forearm_length,
                                radians=True)

            delta_thetas = learning_rate * np.linalg.inv(J) @ error
            thetas += delta_thetas
            # prevent phase jumps due to large errors
            thetas = PlanarArm.circ_values(thetas, radians=True)

        return thetas

    @staticmethod
    def random_theta(return_radians=True,
                     clip_borders_lower: float = 0.0,
                     clip_borders_upper: float = 0.0):
        """
        Returns random joint angles within the limits.
        """
        theta1 = np.random.uniform(PlanarArm.l_upper_arm_limit + clip_borders_lower, PlanarArm.u_upper_arm_limit - clip_borders_upper)
        theta2 = np.random.uniform(PlanarArm.l_forearm_limit + clip_borders_lower, PlanarArm.u_forearm_limit - clip_borders_upper)

        if return_radians:
            return np.array((theta1, theta2))
        else:
            return np.degrees((theta1, theta2))

    @staticmethod
    def random_position(arm: str):
        """
        Returns random position within the joint limits.
        """
        new_theta = PlanarArm.random_theta()
        return PlanarArm.forward_kinematics(arm=arm, thetas=new_theta, radians=True)[:, -1]

    @staticmethod
    def __cos_space(start: float | np.ndarray, stop: float | np.ndarray, num: int):
        """
        For the calculation of gradients and trajectories. Derivation of this function is sin(x),
        so that the maximal change in the middle of the trajectory.
        """

        if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
            if not start.size == stop.size:
                raise ValueError('Start and stop vector must have the same dimensions.')

        # calc changes
        offset = stop - start

        # function to modulate the movement.
        if isinstance(start, np.ndarray):
            x_lim = np.repeat(np.pi, repeats=start.size)
        else:
            x_lim = np.pi

        x = - np.cos(np.linspace(0, x_lim, num, endpoint=True)) + 1.0
        x /= np.amax(x)

        # linear space
        y = np.linspace(0, offset, num, endpoint=True)

        return start + x * y

    def reset_all(self):
        """Reset position to default and delete trajectories"""
        self.__init__(init_thetas=self.trajectory[0],
                      arm=self.arm,
                      radians=True)

    def change_angle(self,
                     new_thetas: np.ndarray,
                     num_iterations: int = 100,
                     radians: bool = False,
                     break_at: None | int = None):
        """
        Change the joint angle of one arm to a new joint angle.
        """

        new_thetas = self.check_values(new_thetas, radians=radians)
        trajectory = self.__cos_space(start=self.angles, stop=new_thetas, num=num_iterations)

        for j, delta_theta in enumerate(trajectory):
            self.trajectory.append(delta_theta - self.trajectory[-1])

            if break_at == j:
                break

        # set current angle to the new thetas
        self.angles = self.trajectory[-1]
        self.update_end_effector()

    def move_to_position(self,
                         end_effector: np.ndarray,
                         num_iterations: int = 100):
        """
        Move to a certain coordinate within the peripersonal space.
        """

        new_thetas_to_position = self.inverse_kinematics(arm=self.arm, end_effector=end_effector,
                                                         starting_angles=self.angles, radians=True)
        self.change_angle(new_thetas=new_thetas_to_position, num_iterations=num_iterations, radians=True)

    def wait(self, time_steps: int):
        for t in range(time_steps):
            self.trajectory.append(self.angles)

    def save_state(self, data_name: str = None):
        import datetime

        d = {
            'trajectory': self.trajectory,
            'arm': self.arm,
        }

        df = pd.DataFrame(d)

        if data_name is not None:
            folder, _ = os.path.split(data_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
        else:
            # get current date
            current_date = datetime.date.today()
            data_name = "PlanarArm_" + current_date.strftime('%Y%m%d')

        df.to_csv(data_name + '.csv', index=False)

    def import_state(self, file: str):
        df = pd.read_csv(file, sep=',')

        # convert type back to np.ndarray because pandas imports them as strings...
        regex_magic = lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=' ', dtype=float)
        self.arm = df['arm'].tolist()[0]
        self.trajectory = df['trajectory'].apply(regex_magic).tolist()[-1]
        self.angles = self.trajectory[-1]
        self.update_end_effector()

    # Functions for visualisation
    def plot_current_position(self, plot_name=None, fig_size=(12, 8)):

        coordinates = PlanarArm.forward_kinematics(arm=self.arm, thetas=self.angles, radians=True)

        fig, ax = plt.subplots(figsize=fig_size)

        ax.plot(coordinates[0, :], coordinates[1, :], 'b')

        ax.set_xlabel('x in [mm]')
        ax.set_ylabel('y in [mm]')

        ax.set_xlim(PlanarArm.x_limits)
        ax.set_ylim(PlanarArm.y_limits)

        # save
        if plot_name is not None:
            folder, _ = os.path.split(plot_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(plot_name)

        plt.show()

    def plot_trajectory(self, fig_size=(12, 8),
                        points: list | tuple | None = None,
                        save_name: str = None,
                        frames_per_sec: int = 10,
                        turn_off_axis: bool = False):

        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        init_t = 0
        num_t = len(self.trajectory)

        coordinates = []
        for i_traj in range(num_t):
            coordinates.append(PlanarArm.forward_kinematics(arm=self.arm,
                                                            thetas=self.trajectory_thetas_left[i_traj],
                                                            radians=True))

        fig, ax = plt.subplots(figsize=fig_size)

        if turn_off_axis:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel('x in [mm]')
            ax.set_ylabel('y in [mm]')

        ax.set_xlim(PlanarArm.x_limits)
        ax.set_ylim(PlanarArm.y_limits)

        line, = ax.plot(coordinates[init_t][0, :], coordinates[init_t][1, :], 'b')

        if points is not None:
            for point in points:
                ax.scatter(point[0], point[1], marker='+')

        val_max = num_t - 1

        if save_name is None:

            ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
            time_slider = Slider(
                ax=ax_slider,
                label='n iteration',
                valmin=0,
                valmax=val_max,
                valinit=0,
            )

            def update(val):
                t = int(time_slider.val)
                line.set_data(coordinates[t][0, :], coordinates[t][1, :])
                time_slider.valtext.set_text(t)

            time_slider.on_changed(update)

            plt.show()
        else:
            def animate(t):
                line.set_data(coordinates[t][0, :], coordinates[t][1, :])
                return line

            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, val_max))

            if save_name[-3:] == 'mp4':
                writer = animation.FFMpegWriter(fps=frames_per_sec)
            else:
                writer = animation.PillowWriter(fps=frames_per_sec)

            ani.save(save_name, writer=writer)
            plt.close(fig)

    @staticmethod
    def calc_motor_vector(init_pos: np.ndarray[float, float], end_pos: np.ndarray[float, float],
                          arm: str, input_theta: bool = False, theta_radians: bool = False):

        if input_theta:
            init_pos = PlanarArm.forward_kinematics(arm=arm, thetas=init_pos, radians=theta_radians)[:, -1]

        diff_vector = end_pos - init_pos
        angle = np.degrees(np.arctan2(diff_vector[1], diff_vector[0])) % 360
        norm = np.linalg.norm(diff_vector)

        return angle, norm, diff_vector

    @staticmethod
    def calc_position_from_motor_vector(init_pos: np.ndarray[float, float],
                                        angle: float,
                                        norm: float,
                                        radians: bool = False):

        x, y = init_pos

        if not radians:
            angle = np.radians(angle)

        new_position = np.array((
            norm * np.cos(angle) + x,
            norm * np.sin(angle) + y
        ))

        return new_position
