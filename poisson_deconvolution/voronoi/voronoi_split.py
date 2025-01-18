import json
import numpy as np
import matplotlib.pyplot as plt

from poisson_deconvolution.microscopy.experiment import generate_bins_loc


class DataSplit:
    def __init__(
        self,
        data: np.ndarray,
        min_pos: np.ndarray,
        max_pos: np.ndarray,
        id: int,
        n_bins: tuple[int],
    ) -> None:
        self.data = data
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.id = id
        self.n_bins = n_bins

    @property
    def split_scale(self):
        return max(self.max_pos - self.min_pos + 1) / max(self.n_bins)

    def plot(self):
        n = max(self.n_bins)
        x1, y1 = self.min_pos / n
        x2, y2 = (self.max_pos + 1) / n
        plt.imshow(
            self.data.T,
            origin="lower",
            extent=[x1, x2, y1, y2],
            cmap="binary",
        )
        plt.title(f"Split {self.id}")

    def copy(self):
        return DataSplit(
            self.data.copy(),
            self.min_pos.copy(),
            self.max_pos.copy(),
            self.id,
            self.n_bins,
        )


class VoronoiSplit:
    @classmethod
    def from_path(cls, path: str):
        with open(path, "r") as file:
            raw = json.load(file)

        return cls.from_json(raw)

    @classmethod
    def from_json(cls, raw: dict):
        nodes = np.array(raw["nodes"])
        delta = raw["delta"]
        n_bins = raw["n_bins"]
        components = [np.array(component) for component in raw["components"]]
        split = np.array(raw["split"])
        return cls(nodes, delta, n_bins, components, split)

    @classmethod
    def empty(cls, nodes: np.ndarray, n_bins: tuple[int] | int):
        return cls(nodes, None, n_bins, [0], 0)

    def __init__(
        self,
        nodes: np.ndarray,
        delta: float,
        n_bins: tuple[int] | int,
        components: list[np.ndarray] = None,
        split: np.ndarray = None,
    ) -> None:
        self.nodes = nodes.copy()
        self.delta = delta
        self.n_bins = (n_bins,) * 2 if isinstance(n_bins, int) else n_bins
        if components is None:
            self.components = connected_components(nodes, delta)
        else:
            self.components = components
        if split is None:
            self.split = self.split_grid()
        else:
            self.split = split

    @property
    def n_components(self):
        return len(self.components)

    def split_grid(self):
        grid = generate_bins_loc(self.n_bins)
        components = self.components
        split_grid = -np.ones(np.prod(self.n_bins))
        for i, node in enumerate(grid):
            min_dist = np.inf
            for j, component in enumerate(components):
                dist = np.min(np.linalg.norm(component - node, axis=1))
                if dist < min_dist:
                    min_dist = dist
                    split_grid[i] = j

        return split_grid.reshape(self.n_bins)

    def split_data(self, data: np.ndarray, id: int) -> DataSplit:
        masked, data_mask = self._masked_data(data, id)
        pos = np.where(data_mask)
        min_pos = np.min(pos, axis=1)
        max_pos = np.max(pos, axis=1)
        split = masked[min_pos[0] : max_pos[0] + 1, min_pos[1] : max_pos[1] + 1]

        return DataSplit(split, min_pos, max_pos, id, self.n_bins)

    def _masked_data(self, data: np.ndarray, id: int) -> np.ndarray:
        data_mask = self.split == id
        return np.ma.array(data, mask=~data_mask, fill_value=0).filled(), data_mask

    def to_json(self):
        raw = {
            "nodes": self.nodes.tolist(),
            "delta": self.delta,
            "n_bins": self.n_bins,
            "components": [component.tolist() for component in self.components],
            "split": self.split.tolist(),
        }
        return raw

    def dump(self, path: str):
        with open(path, "w") as file:
            json.dump(self.to_json(), file)

    def plot_split(self, cmap="tab20", components=False, **kwargs):
        plt.imshow(
            self.split.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, **kwargs
        )
        if components:
            self.plot_components()

    def plot_components(self):
        for i, cluster in enumerate(self.components):
            plt.scatter(
                cluster[:, 0],
                cluster[:, 1],
                label=f"Cluster {i}",
                marker=f"${i}$",
            )
        plt.xlim(0, 1)
        plt.ylim(0, 1)


def connected_components(nodes: np.ndarray, delta: float) -> list[np.ndarray]:
    """
    Connected components of a graph with nodes being connected if their distance is less than delta.
    """
    assert delta > 0, "Delta must be positive."

    connections = [
        [i for i in range(len(nodes)) if np.linalg.norm(node - nodes[i]) <= delta]
        for node in nodes
    ]
    ids = set(range(len(nodes)))
    components = []
    while not len(ids) == 0:
        id = ids.pop()
        component = {id}
        look_ids = {id}
        while not len(look_ids) == 0:
            look_id = look_ids.pop()
            connected = set(connections[look_id])
            look_ids.update(connected - component)
            ids = ids - connected
            component = component | connected
        components.append(np.array([nodes[i] for i in component]))

    return components
