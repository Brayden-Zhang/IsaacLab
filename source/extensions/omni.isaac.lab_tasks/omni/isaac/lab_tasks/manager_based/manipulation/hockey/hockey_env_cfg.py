# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer import OffsetCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
# from omni.isaac.lab_tasks.manager_based.manipulation.hockey.hockey_env_cfg import ActionsCfg, CommandsCfg

from . import mdp

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class HockeySceneCfg(InteractiveSceneCfg):
    """Configuration for the hockey scene with a Franka Panda robot, hockey stick, puck, and net."""
    
    # Franka Panda robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/FrankaPanda",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaPanda/franka_panda.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # # Hockey stick (attached to the robot's end-effector)
    # hockey_stick = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/HockeyStick",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ASSETS_DATA_DIR}/objects/HockeyStick/hockey_stick.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.0),  # This will be relative to the robot's end-effector
    #         rot=(0.0, 0.0, 0.0, 1.0),
    #     ),
    # )

    # # Hockey puck
    # hockey_puck = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/HockeyPuck",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ASSETS_DATA_DIR}/objects/HockeyPuck/hockey_puck.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.5, 0.0, 0.0),  # Starting position of the puck
    #     ),
    # )

    # Hockey net
    hockey_net = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HockeyNet",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ASSETS_DATA_DIR}/objects/HockeyNet/hockey_net.usd",
            usd_path = f"Desktop/local-scratch/IsaacLab/source/extensions/omni.isaac.assets/data/objects/hockey_net/net.usd", 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.0),  # Position of the hockey net
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # # Frame definitions for the hockey stick end
    # hockey_stick_frame = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/HockeyStick",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HockeyStickFrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/HockeyStick/blade",
    #             name="stick_blade",
    #             offset=OffsetCfg(
    #                 pos=(0.0, 0.0, 0.0),
    #                 rot=(0.0, 0.0, 0.0, 1.0),
    #             ),
    #         ),
    #     ],
    # )

    # Retain other necessary elements like plane and light
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

# Update the relevant parts of the MDP settings

# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         puck_position = ObsTerm(func=mdp.get_puck_position)
#         stick_blade_position = ObsTerm(func=mdp.get_stick_blade_position)
#         net_position = ObsTerm(func=mdp.get_net_position)

#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()

# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # Approach the puck
#     approach_stick_puck = RewTerm(func=mdp.approach_stick_puck, weight=2.0, params={"threshold": 0.1})
    
#     # Hit the puck
#     hit_puck = RewTerm(func=mdp.hit_puck, weight=5.0)
    
#     # Puck moves towards the net
#     puck_towards_net = RewTerm(func=mdp.puck_towards_net, weight=3.0)
    
#     # Puck enters the net
#     puck_in_net = RewTerm(func=mdp.puck_in_net, weight=10.0)

#     # Penalize actions for stability
#     action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
#     joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

@configclass
class HockeyEnvCfg(ManagerBasedRLEnvCfg):   
    """Configuration for the hockey environment."""

    # Scene settings
    scene: HockeySceneCfg = HockeySceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    # observations: ObservationsCfg = ObservationsCfg()
    # actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    # rewards: RewardsCfg = RewardsCfg()
    # terminations: TerminationsCfg = TerminationsCfg()
    # events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 15.0
        self.viewer.eye = (-3.0, 3.0, 3.0)
        self.viewer.lookat = (1.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
