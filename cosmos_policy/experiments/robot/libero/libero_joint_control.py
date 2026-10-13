"""Temporary joint-trajectory execution inside an OSC-controlled LIBERO episode."""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation

from memory_system.execute.curobo_trajectory import JointTrajectoryPlan

logger = logging.getLogger(__name__)


def step_correction_controller(controller: Any, observation: dict[str, Any]) -> np.ndarray:
    """Run either the new full-observation controller or a legacy EE controller."""
    if getattr(controller, "requires_observation", False):
        return controller.step(observation)
    current = np.concatenate(
        [
            observation["robot0_eef_pos"],
            Rotation.from_quat(observation["robot0_eef_quat"]).as_rotvec(),
        ]
    ).astype(np.float32)
    return controller.step(current)


class LiberoJointTrajectoryController:
    """Track a timed cuRobo plan, then restore the episode's OSC controller."""

    requires_observation = True

    def __init__(
        self,
        env: Any,
        plan: JointTrajectoryPlan,
        target_ee_states: np.ndarray,
        gripper_command: float,
        max_steps: int = 64,
        position_tolerance: float = 0.005,
        rotation_tolerance: float = 0.02,
        tracking_tolerance: float = 0.08,
        advance_tolerance: float = 0.07,
        stable_steps: int = 2,
        joint_kp: float = 200.0,
        time_scale: float = 2.0,
    ) -> None:
        self.env = env
        self.base_env = env.env
        self.robot = env.robots[0]
        self.plan = plan
        self.target = np.asarray(target_ee_states, dtype=np.float64).reshape(6)
        self.gripper_command = float(gripper_command)
        self.max_steps = int(max_steps)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.tracking_tolerance = float(tracking_tolerance)
        self.advance_tolerance = float(advance_tolerance)
        self.stable_steps = int(stable_steps)
        self.joint_kp = float(joint_kp)
        self.time_scale = float(time_scale)
        if self.time_scale < 1.0:
            raise ValueError("time_scale must be at least 1.0")
        self.control_dt = float(self.base_env.control_timestep)
        sampled = plan.sample_positions(self.control_dt / self.time_scale)
        self.references = self._densify(sampled, max_joint_step=0.04)
        self.planned_steps = len(self.references) - 1
        self.step_count = 0
        self.reference_index = 0
        self.last_reference: np.ndarray | None = None
        self.position_error = float("inf")
        self.rotation_error = float("inf")
        self.tracking_error = 0.0
        self._stable_count = 0
        self._status = "ACTIVE"
        self._closed = False
        self._switched = False
        self._osc_controller = self.robot.controller
        self._osc_config = copy.copy(self.robot.controller_config)
        self._osc_action_dim = int(self.base_env.action_dim)

        if self.references.shape[1] != 7:
            self._status = f"INVALID_DOF:{self.references.shape[1]}"
            return
        if self.planned_steps + self.stable_steps > self.max_steps:
            self._status = "DEADLINE_INFEASIBLE"
            return
        max_delta = np.max(np.abs(np.diff(self.references, axis=0)), axis=0)
        if np.any(max_delta > 0.20):
            self._status = "REFERENCE_STEP_TOO_LARGE"
            return
        self._delta_limit = np.maximum(0.05, max_delta * 1.05)
        self._switch_to_joint_control()

    @staticmethod
    def _densify(positions: np.ndarray, max_joint_step: float) -> np.ndarray:
        from memory.execute.surface_obstacles import densify_joint_path

        return densify_joint_path(positions, max_joint_step)

    def _switch_to_joint_control(self) -> None:
        config = suite.load_controller_config(default_controller="JOINT_POSITION")
        config["output_max"] = self._delta_limit.tolist()
        config["output_min"] = (-self._delta_limit).tolist()
        config["kp"] = self.joint_kp
        self.robot.controller_config = config
        self.robot._load_controller()
        self.robot.controller.update_base_pose(self.robot.base_pos, self.robot.base_ori)
        self.robot.controller.reset_goal()
        self.base_env._action_dim = self.robot.action_dim
        self._switched = True
        logger.info(
            "LIBERO joint trajectory: %.3fs, %d planned steps, %d-step budget",
            self.plan.motion_time * self.time_scale,
            self.planned_steps,
            self.max_steps,
        )

    def _hold_action(self) -> np.ndarray:
        action_dim = self.robot.action_dim if self._switched else self._osc_action_dim
        action = np.zeros(action_dim, dtype=np.float32)
        action[-1] = self.gripper_command
        return action

    @staticmethod
    def _current_ee(observation: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [
                observation["robot0_eef_pos"],
                Rotation.from_quat(observation["robot0_eef_quat"]).as_rotvec(),
            ]
        ).astype(np.float64)

    def observe(self, observation: dict[str, Any]) -> None:
        if self.finished:
            return
        q_actual = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
        if self.last_reference is not None:
            self.tracking_error = float(np.max(np.abs(q_actual - self.last_reference)))
            if self.tracking_error > self.tracking_tolerance:
                self._status = f"TRACKING_ERROR:{self.tracking_error:.5f}"
                return
        current = self._current_ee(observation)
        self.position_error = float(np.linalg.norm(self.target[:3] - current[:3]))
        rotation_error = (
            Rotation.from_rotvec(self.target[3:])
            * Rotation.from_rotvec(current[3:]).inv()
        ).magnitude()
        self.rotation_error = float(rotation_error)
        at_end = self.reference_index == len(self.references) - 1
        within_goal = (
            self.position_error <= self.position_tolerance
            and self.rotation_error <= self.rotation_tolerance
        )
        self._stable_count = self._stable_count + 1 if at_end and within_goal else 0
        deadline_at_goal = self.step_count >= self.max_steps and at_end and within_goal
        if self._stable_count >= self.stable_steps or deadline_at_goal:
            self._status = "CONVERGED"

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        if self.finished:
            return self._hold_action()
        if self.last_reference is None:
            self.observe(observation)
        if self.finished:
            return self._hold_action()
        if self.step_count >= self.max_steps:
            self._status = "GOAL_NOT_CONVERGED"
            return self._hold_action()
        q_actual = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
        next_index = min(self.reference_index + 1, len(self.references) - 1)
        next_reference = self.references[next_index]
        if np.max(np.abs(next_reference - q_actual)) <= self.advance_tolerance:
            self.reference_index = next_index
        reference = self.references[self.reference_index]
        arm_action = np.clip(
            (reference - q_actual) / self._delta_limit, -1.0, 1.0
        )
        action = np.zeros(self.robot.action_dim, dtype=np.float32)
        action[:7] = arm_action.astype(np.float32)
        action[-1] = self.gripper_command
        self.last_reference = reference
        self.step_count += 1
        return action

    @property
    def finished(self) -> bool:
        return self._status != "ACTIVE"

    @property
    def converged(self) -> bool:
        return self._status == "CONVERGED"

    @property
    def status(self) -> str:
        return self._status

    def close(self) -> None:
        if self._closed:
            return
        if self._status == "ACTIVE":
            self._status = "GOAL_NOT_CONVERGED"
        if self._switched:
            self.robot.controller = self._osc_controller
            self.robot.controller_config = self._osc_config
            self.robot.controller.update()
            self.robot.controller.reset_goal()
            self.base_env._action_dim = self._osc_action_dim
        self._closed = True
        logger.info(
            "LIBERO joint trajectory finished: status=%s steps=%d ref=%d/%d pos=%.6fm rot=%.6frad",
            self._status,
            self.step_count,
            self.reference_index,
            len(self.references) - 1,
            self.position_error,
            self.rotation_error,
        )
