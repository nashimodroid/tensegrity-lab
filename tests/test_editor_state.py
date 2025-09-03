from tensegritylab.editor_state import add_strut, edit_strut, delete_strut


def test_round_trip_add_edit_delete():
    struts = []
    # add
    s1 = add_strut(struts, (0, 0, 0), (1, 0, 0))
    assert struts == []  # original not mutated
    assert s1 == [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]

    # edit
    s2 = edit_strut(s1, 0, (0, 0, 0), (1, 1, 1))
    assert s1 == [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
    assert s2 == [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]

    # delete
    s3 = delete_strut(s2, 0)
    assert s3 == []
