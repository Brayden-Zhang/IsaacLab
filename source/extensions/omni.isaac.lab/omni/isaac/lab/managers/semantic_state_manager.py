import torch
from typing import Callable, Sequence, Union

from omni.isaac.orbit.objects.articulated import ArticulatedObject
from omni.isaac.orbit.objects.rigid import RigidObject


def bind(instance: Union[RigidObject, ArticulatedObject], method: Callable[..., None]):
    """
    Binds a method to an instance.
    Args:
        instance: The instance to bind the method to. Usually (RigidObject, ArticulatedObject)
        method: The method to bind. (one of the functions below (add_semantic... etc))
    Returns:
        The bound method.
    """

    def binding_scope_fn(*args, **kwargs) -> Callable[..., None]:
        return method(instance, *args, **kwargs)

    return binding_scope_fn


def add_semantics(self, semantic_obj: "SemanticClass"):
    """
    Adds a semantic object to the semantic list of the instance.
    Args:
        self: The instance.
        semantic_obj: The semantic object to add.
    """
    self.semantic_list.append(semantic_obj)


def reset_all(self, env_ids: Sequence[int]):
    """
    Resets all semantic objects in the instance's semantic list with the given environment IDs.
    Args:
        self: The instance.
        env_ids: The environment IDs.
    """
    for i in self.semantic_list:
        i.reset(env_ids)


def initialize_all(self, path: str):
    """
    Initializes the instance and all semantic objects in its semantic list with the given path.
    Args:
        self: The instance.
        path: The path for initialization.
    """
    self.initialize(path)
    for i in self.semantic_list:
        i.init()


def update_all(self, dt: float):
    """
    Updates the instance's buffers and all semantic objects in its semantic list with the given delta time.
    Args:
        self: The instance.
        dt: The delta time.
    """
    self.update_buffers(dt)
    for i in self.semantic_list:
        i.update()


def initialize_semantics(obj: Union[RigidObject, ArticulatedObject]):
    """
    Initializes the semantics for an object by binding necessary methods and initializing the semantic list.
    Usage:
        .. code-block:: python
            self.cabinets = ArticulatedObject(cfg=self.cfg.cabinet)
            self.cube = RigidObject(cfg=self.cfg.cube)
            initialize_semantics(self.cabinets)
            initialize_semantics(self.cube)
            cube.add_semantics(BooleanSemantic(env))
    Args:
        obj: The object to initialize semantics for (RigidObject, ArticulatedObject)
    """
    obj.add_semantics = bind(obj, add_semantics)
    obj.initialize_all = bind(obj, initialize_all)
    obj.reset_all = bind(obj, reset_all)
    obj.update_all = bind(obj, update_all)
    obj.semantic_list = []


class SemanticClass:
    """Base class for handling semantic states.
    The Semantic States are added to an object with the following functions
    1. update
    2. reset
    3. initialize
    These functions will be called by the manager that the semantic is added to.
    Note: This is a abstract class and should not be used on it's own.
    """

    def __init__(self, env, cfg = None):
        """
        Initializes a semantic object with the given environment.
        Args:
            env: The environment for the semantic object.
        """
        self._env = env
        self.state = None
        self.cfg = cfg 

    def init(self):
        """
        Initializes the semantic object.
        """
        pass

    def reset(self, env_ids: Sequence[int]):
        """
        Resets the semantic object with the given environment IDs.
        Args:
            env_ids: The environment IDs.
        """
        pass

    def update(self):
        """
        Updates the semantic object.
        """
        pass

    def get_state(self) -> torch.Tensor:
        """
        Gets the state of the semantic object.
        Returns:
            The state of the semantic object.
        """
        return self.state