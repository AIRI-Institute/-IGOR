import numpy as np
import re
from typing import List, Tuple
import random
import copy

DIRECTION = {(1, 2, 3): 'southeast', (3, 2, 1): 'skyeast', (1, 3, 2): 'eastsouth', (3, 1, 2): 'skysouth',
             (2, 3, 1): 'eastsky', (2, 1, 3): 'southsky'}

COLOR = {
    'blue': 1,
    'green': 2,
    'red': 3,
    'orange': 4,
    'purple': 5,
    'yellow': 6
}

DIRECTION_LIST = list(DIRECTION.values())
DIRECTION_SORT_KEYS = list(DIRECTION.keys())
DIRECTION_INV = dict(zip(DIRECTION_LIST, DIRECTION_SORT_KEYS))
COLOR_INV = dict(zip(COLOR.values(), COLOR.keys()))

class PrimitiveError(Exception):
    """Exception raised for errors in the initialization of a Primitive object."""

    def __init__(self, message="Invalid primitive"):
        self.message = message
        super().__init__(self.message)
class Primitive:
    def __init__(self,
                 position: Tuple[int, int, int] = None,
                 size: Tuple[int, int, int] = None,
                 direction: str = None,
                 color: str = None,
                 color_digit: int = None,
                 string_instance: str = None):

        if string_instance is not None:
            self._load_from_str(string_instance)
        else:
            self.position = position
            self.size = size
            self.direction = direction
            self.color = color if color else COLOR_INV[color_digit]

    def __repr__(self) -> str:
        return f"{list(self.position)}, {list(self.size)}, '{self.direction}', '{self.color}'"
        
    def _load_from_str(self, string_instance):
        spl = re.findall(r'[\w1-9]+', string_instance)
        if len(spl) == 8:
            self.position = (int(spl[0]), int(spl[1]), int(spl[2]))
            self.size = (int(spl[3]), int(spl[4]), int(spl[5]))
            self.direction = spl[6]
            self.color = spl[7]
        else:
            raise PrimitiveError()

    def to_base(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], str]:
        inv_direction = DIRECTION_INV[self.direction]
        x, y, z = inv_direction
        new_sizes = self.size[x - 1], self.size[y - 1], self.size[z - 1]
        return (self.position, new_sizes, self.color)

    def to_grid(self) -> np.ndarray:
        grid = np.zeros((9, 11, 11))
        z, x, y = self.position
        try:
            ids = DIRECTION_INV[self.direction]
        except Exception as e:
            return grid
        dz, dx, dy = self.size[ids[0]-1], self.size[ids[1]-1], self.size[ids[2]-1]
        try:
            color = COLOR[self.color]
        except Exception:
            return grid
        grid[z: z+dz, x: x+dx, y: y+dy] = color
        return grid

    def augment(self, probabilities=None):

        new_primitive = copy.deepcopy(self)

        if probabilities is None:
            probabilities = [0.25, 0.25, 0.25, 0.25]  # Равные вероятности по умолчанию
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        choice = random.choices(['position', 'size', 'color', 'direction'], weights=probabilities, k=1)[0]

        if choice == 'position':
            # Изменяем позицию на 1 в случайном направлении
            axis = random.choices([0,1,2], weights=[0.2, 0.4, 0.4], k=1)[0]
            change = random.choice([-1, 2])
            
            if self.position[axis] <= 1:
                change = abs(change)
                
            new_primitive.position = tuple(
                p + (change if i == axis else 0) for i, p in enumerate(new_primitive.position)
            )

        elif choice == 'size':
            axis = random.randint(0, 2)
            change = random.choice([-1, 2])

            if self.size[axis] <= 1:
                change = abs(change)
            new_primitive.size = tuple(
                s + (change if i == axis else 0) for i, s in enumerate(new_primitive.size)
            )

        elif choice == 'color':
            new_color = random.choice(list(COLOR.keys()))
            while new_color == new_primitive.color:
                new_color = random.choice(list(COLOR.keys()))
            new_primitive.color = new_color

        elif choice == 'direction':
            new_direction = random.choice(DIRECTION_LIST)
            while new_direction == new_primitive.direction:
                new_direction = random.choice(DIRECTION_LIST)
            new_primitive.direction = new_direction

        return new_primitive
def centralize(primitives: List[Primitive]):
    dx = primitives[0].position[1] - 5
    dy = primitives[0].position[2] - 5
    for i, prim in enumerate(primitives):
        primitives[i].position[1] -= dx
        primitives[i].position[2] -= dy
    return primitives

def to_str(primitives: List[Primitive]) -> str:
    return ' && '.join(str(primitive) for primitive in primitives)

def from_str(s: str) -> List[Primitive]:
    s = s.split('&&')
    prims = []
    for prim in s:
         try:
             prims.append(Primitive(string_instance=prim))
         except PrimitiveError as e:
             print(e)
    return prims

def to_grid(primitives: List[Primitive]) -> np.ndarray:
    grid = np.zeros((9, 11, 11))
    for prim in primitives:
        grid += prim.to_grid()
    return grid

def bbox(grid: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
    bgrid = np.zeros((9, *bounds))
    try:
        xmin = grid.nonzero()[1][0]
        ymin = grid.nonzero()[2][0]
    except IndexError:
        xmin = ymin = 0

    cut_grid = grid[:, xmin: xmin + bounds[0], ymin: ymin + bounds[1]]
    bgrid[:, :cut_grid.shape[1], :cut_grid.shape[2]] = cut_grid
    return bgrid
