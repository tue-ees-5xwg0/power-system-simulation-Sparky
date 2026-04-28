import GraphTools.graph_processing as gp


def test_valid_input():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )
    assert processor is not None


def test_invalid_vertex_ids():
    vertex_ids = [1, 2, 2]  # Duplicate vertex ID
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected IDNotUniqueError")
    except gp.IDNotUniqueError:
        pass


def test_invalid_edge_ids():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 1]  # Duplicate edge ID
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected IDNotUniqueError")
    except gp.IDNotUniqueError:
        pass


def test_invalid_edge_vertex_id_pairs():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 4), (2, 3)]  # Invalid vertex ID in edge_vertex_id_pairs
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected IDNotFoundError")
    except gp.IDNotFoundError:
        pass


def test_invalid_edge_enabled_length():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True]  # Length does not match edge_ids
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected InputLengthDoesNotMatchError")
    except gp.InputLengthDoesNotMatchError:
        pass


def test_invalid_edge_vertex_id_pairs_length():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2)]  # Length does not match edge_ids
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected InputLengthDoesNotMatchError")
    except gp.InputLengthDoesNotMatchError:
        pass


def test_invalid_source_vertex_id():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 4  # Invalid source id

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected IDNotFoundError")
    except gp.IDNotFoundError:
        pass


def test_graph_not_fully_connected():
    vertex_ids = [1, 2, 3, 4]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (3, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected GraphNotFullyConnectedError")
    except gp.GraphNotFullyConnectedError:
        pass


def test_graph_cycle_error():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2, 3]
    edge_vertex_id_pairs = [(1, 2), (2, 3), (1, 3)]
    edge_enabled = [True, True, True]
    source_vertex_id = 1

    try:
        gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        AssertionError("Expected GraphCycleError")
    except gp.GraphCycleError:
        pass


def test_disabled_edge_ignored_in_graph_construction():
    vertex_ids = [1, 2, 3, 4]
    edge_ids = [1, 2, 3, 4]
    edge_vertex_id_pairs = [(1, 2), (2, 3), (3, 4), (4, 1)]
    edge_enabled = [True, True, True, False]
    source_vertex_id = 1

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )

    edge_ids_in_graph = {data["edge_id"] for _, _, data in processor.graph.edges(data=True)}
    assert edge_ids_in_graph == {1, 2, 3}


def test_find_downstream_vertices_1():
    vertex_ids = [0, 2, 4]
    edge_ids = [1, 3]
    edge_vertex_id_pairs = [(0, 2), (2, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 0

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )

    assert processor.find_downstream_vertices(1) == [2, 4]


def test_find_downstream_vertices_2():
    vertex_ids = [0, 2, 4]
    edge_ids = [1, 3]
    edge_vertex_id_pairs = [(0, 2), (2, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 0

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )

    assert processor.find_downstream_vertices(3) == [4]


def test_find_downstream_vertices_IdNotfound():
    vertex_ids = [0, 2, 4]
    edge_ids = [1, 3]
    edge_vertex_id_pairs = [(0, 2), (2, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 0

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )
    try:
        processor.find_downstream_vertices(5)
        raise AssertionError("Expected IDNotFoundError")
    except gp.IDNotFoundError:
        pass


def test_find_downstream_vertices_disabled():
    vertex_ids = [0, 2, 4, 5]
    edge_ids = [1, 3, 4, 5]
    edge_vertex_id_pairs = [(0, 2), (2, 4), (2, 5), (4, 5)]
    edge_enabled = [True, False, True, True]
    source_vertex_id = 0

    processor = gp.GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_id_pairs,
        edge_enabled,
        source_vertex_id,
    )

    assert processor.find_downstream_vertices(3) == []
