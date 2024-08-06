class Node:
    """
    Represents a station in a bike sharing system with its demand and status.

    Attributes:
        demand (int): The current demand of the station.
        is_empty (bool): Indicates whether the station is empty.
        is_full (bool): Indicates whether the station is full.
    """

    is_empty: bool
    is_full: bool
    demand: int

    def __init__(self, demand: int, is_empty=False, is_full=False) -> None:
        """
        Initializes a Node instance.

        Args:
            demand (int): The initial demand of the station.
            is_empty (bool, optional): The initial empty status of the station. Defaults to False.
            is_full (bool, optional): The initial full status of the station. Defaults to False.
        """
        self.demand = demand
        self.is_empty = is_empty
        self.is_full = is_full

    def get_demand(self) -> int:
        return self.demand

    def set_demand(self, demand: int) -> None:
        self.demand = demand

    def check_is_empty(self) -> bool:
        return self.is_empty

    def set_is_empty(self) -> None:
        """
        Toggles the empty status of the station.
        """
        self.is_empty = not self.is_empty

    def check_is_full(self) -> bool:
        return self.is_full

    def set_is_full(self) -> None:
        """
        Toggles the full status of the station.
        """
        self.is_full = not self.is_full
