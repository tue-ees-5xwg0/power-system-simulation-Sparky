"""
This is a skeleton for the graph processing assignment.

We define a graph processor class with some function skeletons.
"""

from typing import List, Tuple
import networkx as nx

class IDNotFoundError(Exception):
    pass


class InputLengthDoesNotMatchError(Exception):
    pass


class IDNotUniqueError(Exception):
    pass


class GraphNotFullyConnectedError(Exception):
    pass


class GraphCycleError(Exception):
    pass


class EdgeAlreadyDisabledError(Exception):
    pass


class GraphProcessor:
    """
    General documentation of this class.
    You need to describe the purpose of this class and the functions in it.
    We are using an undirected graph in the processor.
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.
        The following conditions should be checked:
            1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
            2. edge_vertex_id_pairs should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            3. edge_vertex_id_pairs should contain valid vertex ids. (IDNotFoundError)
            4. edge_enabled should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
            6. The graph should be fully connected. (GraphNotFullyConnectedError)
            7. The graph should not contain cycles. (GraphCycleError)
        If one certain condition is not satisfied, the error in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        # put your implementation here

        # data sanitazion and validation
        if len(set(vertex_ids)) != len(vertex_ids):
            raise IDNotUniqueError("vertex_ids should be unique.")
        if len(set(edge_ids)) != len(edge_ids):
            raise IDNotUniqueError("edge_ids should be unique.")
        if len(edge_vertex_id_pairs) != len(edge_ids):
            raise InputLengthDoesNotMatchError(
                "edge_vertex_id_pairs should have the same length as edge_ids."
            )
        if len(edge_enabled) != len(edge_ids):
            raise InputLengthDoesNotMatchError(
                "edge_enabled should have the same length as edge_ids."
            )
        if source_vertex_id not in vertex_ids:
            raise IDNotFoundError("source_vertex_id should be a valid vertex id.")
        for vertex_id_pair in edge_vertex_id_pairs:
            if vertex_id_pair[0] not in vertex_ids or vertex_id_pair[1] not in vertex_ids:
                raise IDNotFoundError(
                    "edge_vertex_id_pairs should contain valid vertex ids."
                )
        
        # Build the enabled-edge graph and store edge lookup tables.
        self.vertex_ids = list(vertex_ids)
        self.edge_ids = list(edge_ids)
        self.edge_vertex_id_pairs = list(edge_vertex_id_pairs)
        self.edge_enabled = list(edge_enabled)
        self.source_vertex_id = source_vertex_id

        self.edge_id_to_vertices = dict(zip(edge_ids, edge_vertex_id_pairs))
        self.edge_id_to_enabled = dict(zip(edge_ids, edge_enabled))

        self.graph = nx.Graph()
        self.graph.add_nodes_from(vertex_ids)
        for edge_id, (vertex_a, vertex_b), enabled in zip(
            edge_ids, edge_vertex_id_pairs, edge_enabled
        ):
            if enabled:
                self.graph.add_edge(vertex_a, vertex_b, edge_id=edge_id)

        if not nx.is_connected(self.graph):
            raise GraphNotFullyConnectedError("graph should be fully connected.")
        if not nx.is_tree(self.graph):
            raise GraphCycleError("graph should not contain cycles.")


    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """
        # put your implementation here
        pass

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        # put your implementation here
        pass
