# Joe Nyhan, 30 June 2021
# Array operations, etc., for the simulation.

from hd_params import *


@njit
def checkFloor(grid):
    """
    for a given 1D array, check that every value is above the floor threshold; if not, set it equal to the floor
    """
    for i in range(len(grid)):
        if grid[i] < floor:
            grid[i] = floor

@njit
def initializeEvenVPs(grid, spacing="unstaggered"):
    """
    initializes the virtual points to match the first real gridpoints symmetrically
    """
    # used for alpha, due to our convention of storing values at the gridpoint that 
    # correspond to the cell bondary to its left
    if spacing == "staggered":
        for i in range(NUM_VPOINTS):
            grid[NUM_VPOINTS-1-i] = grid[NUM_VPOINTS+i]
    if spacing == "unstaggered":
        for i in range(NUM_VPOINTS):
            grid[NUM_VPOINTS-1-i] = grid[NUM_VPOINTS+1+i]


@njit
def initializeOddVPs(grid):
    """
    initializes the virtual points to match the first real gridpoints anti-symmetrically
    """
    for i in range(NUM_VPOINTS):
        grid[NUM_VPOINTS-1-i] = -grid[NUM_VPOINTS+i]
