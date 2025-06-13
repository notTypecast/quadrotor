# HMPC Method
Implementation of the HMPC (Hybrid MPC) method. A multi-layer perceptron is used to supplement a dynamic model of a system, enabling the resulting _hybrid model_ to learn via a online appoach and correct model uncertainties.

Specifically, this repository implements the following:
* A simulator for a planar quadrotor system. The planar quadrotor (or 2D quadrotor) is a simplification of the quadrotor system for two dimensions. The simulation only includes gravity, as well as two forces acting on the PQ, one on each rotor, in the direction of the first normal of the PQ.
* A simulator for a quadrotor system. Similarly, the simulator includes gravity, as well as the four rotor speeds. The forces and torques are calculated by multiplying specific constants $K_f$ and $K_t$ with the rotor speeds.
* An optimizer, utlizing the CEM algorithm to control the PQ and lead it to a specified position. This is implemented for the planar quadrotor only.
* An optimizer, utilizing numerical optimization using CasADi to control the quadrotor and lead it to a specified position. This is implemented for both the 2D and the 3D quadrotors.
* A symbolic neural network, created using CasADi.

## Planar Quadrotor
### Simulation
#### Simulator
The planar quadrotor simulator is implemented in `src/sim/PlanarQuadrotor.hpp`. The `PlanarQuadrotor` class initializes a new PQ with the given mass, moment of inertia and length. The PQ's initial position is $(0, 0)$.

Using the `update` method, the system is integrated for a given time step, using the provided controls (forces for each rotor). The controls are considered constant for that time step.

The `get_state` method returns the current state of the PQ. The state is a 6D vector, which consists of the linear ($x$, $y$) and angular ($w$) position, as well as the respective velocities. Since acceleration is constant and directly related to the controls passed at each time step, it is not part of the state.

The `get_sim_time` method returns the total time the simulation has been integrated for. The `get_last_ddq` method returns the acceleration computed during the previous integration step.

#### Visualizer
The visualizer class, located at `src/sim/Visualizer.hpp`, can be used to visualize the PQ in a (linux) command-line enviroment. The `show` method accepts a `PlanarQuadrotor` object, as well as a pair of integers representing the target position ($x_t$, $y_t$). It then prints the equivalent frame on the screen. An arrow is used to represent the PQ, pointing towards the direction perpendicular to the PQ's length. The target position is represented by the letter T.

The `set_message` method can be used to set a specific message to be printed at the bottom of the screen, for every subsequent frame.

### Optimization
#### Dynamic model
To determine the controls required for the PQ to move towards the target position, we use MPC, utilizing the CEM algorithm, implemented in the linked `algevo` library. This repository defines the `ControlIndividual` struct. This consists of a vector of $2h$ values, where $h$ is the horizon, a defined parameter. The struct also defines the method by which each individual is evaluated: the "simulation" is run for $h$ steps and the error is calculated at each step. Consequently, the individual's fitness is relative to the error at each of the $h$ steps.

The algorithm runs iteratively and, once finished, provides the individual with the best overall fitness value. Of this individual, we only use the first two values ($c_{11}$ and $c_{12}$). In this way, the optimizer is able to take into account multiple steps into the future, but we only follow the best initial move, so that we can recalculate the next best move on the next integration step.

In order to run this simulation at each time step, for each individual, we need a model: a method for calculating the accelerations at that time step, based on the input controls. For the PQ, we can easily derive the following forward dynamics model:

$$\Large
\begin{align}
    \ddot{\textbf{q}} &= \begin{bmatrix}
        \frac{-(c_1+c_2)sin(q_3)}{m}\\\\
        \frac{(c_1+c_2)cos(q_3)}{m} - g\\\\
        \frac{(c_2 - c_1)l}{2I}
    \end{bmatrix}
\end{align}
$$

Where:
* $c_1$ and $c_2$ are the provided controls.
* $q_3$ is the angular position of the PQ.
* $g$ is the gravitational acceleration.
* $m$, $l$ and $I$ are the mass, length and moment of inertia of the PQ, respectively.

Since the simulator and the optimizer use the exact same dynamic model, this works perfectly on its own. As long as an appropriate cost function is used, the optimizer is easily able to calculate the required controls to reach the target position.

Next, it is necessary to mimic a situation in which the dynamic model is either not known, or is too complex to account for. In order to do this, there needs to be a mismatch between the actual dynamic model, used by the simulator, and the dynamic model used by the optimizer to find the optimal controls. There are multiple ways to do this, but in this repository, we alter the mass (and subsequently inertia) used by the optimizer.

With a large enough difference between the two gravitational acceleration values, the controls provided by the optimizer result in movement that leads the PQ far from the desired target position.

#### Learned model
Now that the true and used model are different to each other, the predicted state on each time step is different to the true state of the system. This difference can be learned, so that the existing dynamic model can be enhanced to be more accurate to the true model.

More specifically, the input of the learned model needs to be the same as that of the dynamic model: the state of the PQ, as well as the controls used for the next time step. The training target needs to be the difference between the actual acceleration caused and the predicted acceleration of the existing dynamic model.

$$
    \begin{align}
        \textbf{x} &= \begin{bmatrix}
            \textbf{q}\\
            \dot{\textbf{q}}\\
            c_1\\
            c_2
        \end{bmatrix},\
        \textbf{t} = \textbf{a}-\textit{d}(\textbf{x})
    \end{align}
$$

Where:
* $\textbf{x}$ and $\textbf{t}$ are the input and desired output, respectively.
* $\textbf{q}$ is the position vector.
* $\dot{\textbf{q}}$ is the velocity vector.
* $c_1$ and $c_2$ are the controls.
* $\textbf{a}$ is the actual acceleration caused by integrating using these controls.
* $d(\cdot)$ is the dynamic model.

A learned model trained in this way allows us to then express the predicted value for each time step as such:

$$
    \begin{align}
        \ddot{\textbf{q}_p} = \textit{d}(\textbf{x}) + \textit{l}(\textbf{x})
    \end{align}
$$

Where $\textit{l}(\cdot)$ is the learned model.

Note that, as seen in $(1)$, the dynamic model only uses the angular position value, $q_3$. Thus, the entire state is not actually necessary and, in fact, we would likely have better performance if we only passed $q_3$ as an input to our learned model. However, in a real-world scenario, we would usually not be aware of which parameters of the state are or are not useful. Therefore, we pass the entire state vector and assume that any non-useful parameters will end up not contributing to the result.

In this repository, a neural network is used to learn this difference. We use the linked library `simple_nn` to initialize a neural network and train it using an episodic approach. This consists of running the simulation and optimization as-is for `n` steps, which make up an episode. We then train the neural network using all collected data during the episode. Following this, we repeat the process again for `m` episodes, or until the change in error between episodes is smaller than a specific threshold value.

This entire process is repeated for `k` runs. We do this to collect data for multiple runs and get a mean error value per step per episode.

#### Numerical optimization
Instead of using CEM to calculate the optimal controls, there is also the option to use numerical optimization. This is implemented using CasADi. The dynamic model has been implemented symbolically, allowing for optimization of the control forces based on the given dynamics.

Additionally, using the symbolic neural network, a learned model can be trained similarly to the above, to learn differences between the actual and the known dynamics.

## Quadrotor
### Simulation
#### Simulator
The equivalent simulator for the quadrotor is located at `src/sim/Quadrotor.hpp`. This operates similarly to the PQ, with the expected differences found in 3 dimensions. The moment of inertia is represented by a $3x3$ inertia matrix. Optional parameters include the constants $K_f$ and $K_t$.

It should be noted that orientation in 3 dimensions is represented using a unit quaternion. As such, the quadrotor state consists of a 13-dimensional vector, which includes the linear position (size 3), the quaternion (size 4), the linear velocity (size 3) and the angular velocity (size 3). The position and orientation are kept in world frame, but the velocity and angular velocity are kept in body frame.

#### Visualizer
To visualize the quadrotor system's states, we use _matplotlib_. During execution, system state data is written to `src/train/data/quad.txt`. At the same time, this data can be read and visualized by the `src/train/VisualizeQuad.py` script.

The script is also able to recognize certain commands, specifically for setting the target position, as well as the title of the graph.

### Optimization
#### Dynamic model
As mentioned, only numerical optimization is implemented with the 3D quadrotor system. The dynamics for the 3D quadrotor are slightly more complicated than those of the planar quadrotor.

Since our model uses rotor speeds as the control input, those need to be converted to thrust and torque, before they can be used to calculate acceleration. This is generally done by multiplying with constants $K_f$ and $K_t$. As such, to calculate the total thrust and torque, in the body frame, we use the below equations.

$$\Large
    \begin{align}
        \mathbf{F}_b = \mathbf{R}^T \begin{bmatrix}
        0\\
        0\\
        -mg
        \end{bmatrix} + \begin{bmatrix}
        0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0\\
        K_f & K_f & K_f & K_f
        \end{bmatrix}\mathbf{c}
    \end{align}
$$

Where:
* $\mathbf{R}$ is the rotation matrix, derived from the quaternion representing the state of the quadrotor. We transpose this matrix since the gravity vector must be converted from world frame to body frame.
* $\mathbf{c}$ is a vector of size 4, containing the rotor speeds.

$$\Large
    \begin{align}
        \mathbf{\tau}_b = \begin{bmatrix}
        \ell K_f(c_2 - c_4)\\
        \ell K_f(c_3 - c_1)\\
        K_t(c_1 - c_2 + c_3 - c_4)
        \end{bmatrix}
    \end{align}
$$

Where:
* $c_i$ are the rotor speeds.
* $\ell$ is the length from the center of the quadrotor to each rotor.

Once the values for thrust and torque have been calculated, we can calculate the linear and angular acceleration using the following formulas.

$$\Large
    \begin{align}
    \dot{\mathbf{u}}_b = \frac{1}{m}\mathbf{F}_b - \mathbf{\omega}_b \times \mathbf{u}_b
    \\
    \dot{\mathbf{\omega}}_b = I_m^{-1}(\mathbf{\tau}_b - \mathbf{\omega}_b \times I_m \mathbf{\omega}_b)
    \end{align}
$$

Where:
* $\mathbf{u}_b$ is the linear velocity vector.
* $\mathbf{\omega}_b$ is the angular velocity vector.
* $I_m$ is the inertia matrix of the quadrotor.

It should be added that at this point, in the planar quadrotor version, we would simply integrate using the calculated acceleration. In this case, however, there are still two issues. Firstly, both of these accelerations are in the body frame, but the position and orientation are in the world frame. Therefore, after integrating the velocities, we need to convert to world frame in order to integrate the position and orientation. Additionally, our orientation is kept as a quaternion, so we need to update it according to this angular acceleration.

Contrary to the planar quadrotor version, we perform semi-implicit Euler integration. As such, we update the quadrotor velocity first, after which we use the updated velocity to integrate the position for the given timespan.

$$\Large
    \begin{align}
        \mathbf{u}_w = \mathbf{R}\mathbf{u}_b
    \end{align}
$$

To get the world frame linear velocity, all that is needed is to multiply it from the left by the rotation matrix. We then use this velocity to integrate the linear position.

As for the orientation:

$$\Large
    \begin{align}
        \dot{\mathbf{q}} = \frac{1}{2} \begin{bmatrix}
        -x & -y & -z\\
        w & -z & y\\
        z & w & -x\\
        -y & x & w \end{bmatrix} \mathbf{\omega}_b
    \end{align}
$$

Where:
* $q = \begin{bmatrix} w & x & y & z \end{bmatrix}^T$ is the quaternion representing the orientation.

Using $\dot{\mathbf{q}}$, we can integrate the orientation quaternion. Note that the above formula requires that the quaternion be a unit quaternion, but the produced quaternion after integration will not be a unit quaternion. As such, the produced quaternion must then be normalized.

#### Learned model
The learned model works in the same way as in the planar quadrotor version. The only difference is that here, the input is a 17-dimensional vector (13 for the state and 4 for the input controls) and the output is a 6-dimensional vector (3 for linear and 3 for angular acceleration.)