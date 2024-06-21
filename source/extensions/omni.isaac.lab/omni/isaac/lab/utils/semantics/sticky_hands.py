import torch
from omni.isaac.orbit.utils.mdp.semantic_state_manager import SemanticClass
from omni.isaac.contrib_envs.semantics_cfg.semantics_cfg import StickyGrabCfg


class StickyGrab(SemanticClass):
    def __init__(self, env, cfg = StickyGrabCfg()):
        """
        Initialize the StickyGrab class.
        Args:
            env: The environment object.
            obj: The object to grab.
            grip_track: The GripperTracker object.
            threshold: The threshold distance for grabbing.
        """
        super().__init__(env, cfg)
        self.robot = env.robots[cfg.robot]
        self.obj = env.objects[cfg.obj]
        self.threshold = self.cfg.threshold

    def reset(self, env_ids):
        self.state[env_ids] = False

    def magnet_active(self):
        return True

    def update(self):
        in_range = torch.norm(self.robot.data.ee_state_w[:, :3] - self.obj.data.root_pos_w) < self.threshold
        active = self.magnet_active()

        self.state = in_range & active
        for i in range(10):
            print(self.state, in_range, active)

        positions = self.obj.data.root_state_w.clone()
        positions[:, :] = torch.where(
            self.state,
            torch.cat([self.robot.data.ee_state_w[:, :3], positions[:, 3:7], self.robot.data.ee_state_w[:, 7:]], dim=1),
            self.obj.data.root_state_w
        )
        self.obj.set_root_state(positions)

class OT2MagneticAttach(StickyGrab):

    def magnet_active(self):
        return self.robot.data.dof_pos[:, -1] + self.robot.data.dof_pos[:, -2] < 0.07