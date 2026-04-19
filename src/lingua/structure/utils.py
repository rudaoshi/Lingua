"""
utilities for graph manipulation and visualization
"""


import json
from typing import Union

from enum import Enum, auto


class AutoNamedEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 70
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 3
    """Maximum number of items in container that might be put on single line."""

    INDENTATION_CHAR = " "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

        if self.indent is None:
            self.indent = 2

        self.list_nest_level = 0
        self.kwargs = kwargs

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                self.list_nest_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                self.list_nest_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return "{ " + ", ".join(f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()) + " }"
                else:
                    self.indentation_level += 1
                    output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                    self.indentation_level -= 1
                    return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{}"
        elif isinstance(o, float):  # Use scientific notation for floats, where appropiate
            return format(o, "g")
#        elif isinstance(o, str):  # escape newlines
#            o = o.replace("\n", "\\n")
#            return f'"{o}"'
        else:
            return json.dumps(o, **self.kwargs)

    def _put_on_single_line(self, o):
        return self._primitives_only(o) and len(o) <= self.MAX_ITEMS and len(str(o)) - 2 <= self.MAX_WIDTH

    def _primitives_only(self, o: Union[list, tuple, dict]):
        if self.list_nest_level >= 1:
            return True
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        return self.INDENTATION_CHAR*(self.indentation_level*self.indent)


def positions2spans(words):
    """
    check
    """

    if isinstance(words, int):
        words = tuple([words])
    elif isinstance(words, (list, tuple)):
        words = tuple(words)
    else:
        raise Exception("the words must be an int or a list of int/str")

    spans = []
    idx = 0
    while idx < len(words):
        if isinstance(words[idx], str):
            spans.append(words[idx])
            idx += 1
        else:
            start_idx = idx
            while start_idx + 1 < len(words) \
                and (words[start_idx + 1] == words[start_idx] + 1 \
                    or words[start_idx + 1] == words[start_idx]):
                start_idx += 1

            spans.append((words[idx], words[start_idx]))
            idx = start_idx + 1

    return tuple(spans)