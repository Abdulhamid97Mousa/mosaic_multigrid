"""Miscellaneous utilities: front_pos helper and PropertyAlias descriptor."""
from __future__ import annotations

import functools


@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int) -> tuple[int, int]:
    """Get the position in front of an agent given position and direction."""
    from ..core.constants import Direction
    dx, dy = Direction(agent_dir).to_vec()
    return (agent_x + dx, agent_y + dy)


class PropertyAlias(property):
    """
    A class property that is an alias for an attribute's property.

    Instead of writing separate getter/setter, declare::

        x = PropertyAlias('state', 'x')
    """

    def __init__(self, attr_name: str, attr_property_name: str, doc: str = None) -> None:
        prop = lambda obj: getattr(type(getattr(obj, attr_name)), attr_property_name)
        fget = lambda obj: prop(obj).fget(getattr(obj, attr_name))
        fset = lambda obj, value: prop(obj).fset(getattr(obj, attr_name), value)
        fdel = lambda obj: prop(obj).fdel(getattr(obj, attr_name))
        super().__init__(fget, fset, fdel, doc=doc)
        self.__doc__ = doc
