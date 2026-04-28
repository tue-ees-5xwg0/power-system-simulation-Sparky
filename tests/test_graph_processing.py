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
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected IDNotUniqueError"
    except gp.IDNotUniqueError:
        pass

def test_invalid_edge_ids():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 1]  # Duplicate edge ID
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected IDNotUniqueError"
    except gp.IDNotUniqueError:
        pass

def test_invalid_edge_vertex_id_pairs():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 4), (2, 3)]  # Invalid vertex ID in edge_vertex_id_pairs
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected IDNotFoundError"
    except gp.IDNotFoundError:
        pass

def test_invalid_edge_enabled_length():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True]  # Length does not match edge_ids
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected InputLengthDoesNotMatchError"
    except gp.InputLengthDoesNotMatchError:
        pass


def test_invalid_edge_vertex_id_pairs_length():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2)]  # Length does not match edge_ids
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected InputLengthDoesNotMatchError"
    except gp.InputLengthDoesNotMatchError:
        pass


def test_invalid_source_vertex_id():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (2, 3)]
    edge_enabled = [True, True]
    source_vertex_id = 4  # Invalid source id

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected IDNotFoundError"
    except gp.IDNotFoundError:
        pass


def test_graph_not_fully_connected():
    vertex_ids = [1, 2, 3, 4]
    edge_ids = [1, 2]
    edge_vertex_id_pairs = [(1, 2), (3, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected GraphNotFullyConnectedError"
    except gp.GraphNotFullyConnectedError:
        pass


def test_graph_cycle_error():
    vertex_ids = [1, 2, 3]
    edge_ids = [1, 2, 3]
    edge_vertex_id_pairs = [(1, 2), (2, 3), (1, 3)]
    edge_enabled = [True, True, True]
    source_vertex_id = 1

    try:
        processor = gp.GraphProcessor(
            vertex_ids,
            edge_ids,
            edge_vertex_id_pairs,
            edge_enabled,
            source_vertex_id,
        )
        assert False, "Expected GraphCycleError"
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

    edge_ids_in_graph = {
        data["edge_id"] for _, _, data in processor.graph.edges(data=True)
    }
    assert edge_ids_in_graph == {1, 2, 3}