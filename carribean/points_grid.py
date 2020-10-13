
from typing import Callable, Tuple, List, Iterator, Optional

import numpy as np
import pandas as pd

Point = Tuple[int, int]


def four_points_connectivity(point: Point) -> List[Point]:
    x, y = point
    xs = [x - 1, x + 1, x, x]
    ys = [y, y, y - 1, y + 1]
    return list(zip(xs, ys))


def eight_points_connectivity(point: Point) -> List[Point]:
    x, y = point
    result = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
              (x - 1, y), (x + 1, y),
              (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
    return result


class PointsGridGraph:
    """
    Provides some graph abstractions and functionality for 2d points grid based on
    some provided connectivity strategy

    Arguments
    ---------
    input_map:
        input pixel grid
    connectivity_strategy:
        provided connectivity strategy
    """

    def __init__(self, input_map: np.ndarray, connectivity_strategy: Callable[[Point], List[Point]]):
        self._input_map = input_map
        self._visited_map = np.zeros_like(input_map)
        self._connectivity = connectivity_strategy
    
    def is_visited(self, point: Point) -> bool:
        x, y = point
        return 0 != self._visited_map[x, y]  # pylint: disable=invalid-sequence-index
    
    def is_valid_node(self, point: Point) -> bool:
        x, y = point
        return (0 <= x and 0 <= y
                and x < self._input_map.shape[0] and y < self._input_map.shape[1]
                and self._input_map[x, y] > 0)
    
    def __iter__(self) -> Iterator[Point]:
        
        for x in range(self._input_map.shape[0]):
            for y in range(self._input_map.shape[1]):
                yield (x, y)

    def get_connected_points(self, point: Point) -> List[Point]:
        neighbours = self._connectivity(point)
        result = [point for point in neighbours if self.is_valid_node(point) and not self.is_visited(point)]
        return result
    
    def traverse(self, start_point: Point) -> List[Point]:

        next_points = [start_point]
    
        result = []
        while(next_points):
            point = next_points.pop()
            if not self.is_visited(point):
                self._visited_map[point[0], point[1]] = 1 # pylint: disable=unsupported-assignment-operation
                next_points += self.get_connected_points(point)
                result.append(point)
        
        return result
    
    def get_connected_components(self, min_component_size: Optional[int] = None) -> pd.DataFrame:
        result = []
        for point in self:
            if self.is_valid_node(point) and not self.is_visited(point):
                component = self.traverse(point)
                result.append(component)
        
        if min_component_size:
            result = filter(lambda x: min_component_size <= len(x), result)
        result = (np.array(component) for component in result)
        result = [pd.DataFrame({'x': component[:, 0],
                                'y': component[:, 1],
                                'value': self._input_map[component[:, 0], component[:, 1]],
                                'island': i + 1}) for i, component in enumerate(result)]
        result = pd.concat(result)
        return result
