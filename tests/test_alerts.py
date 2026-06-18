from firevision.alerts import PersistentAlert


def test_persistent_alert_rejects_single_frame_false_positive():
    alert = PersistentAlert(window=5, required=3, cooldown=2)
    assert [alert.update(value) for value in [False, True, False, False, False]] == [
        False,
        False,
        False,
        False,
        False,
    ]


def test_persistent_alert_fires_after_required_votes():
    alert = PersistentAlert(window=5, required=3, cooldown=2)
    outputs = [alert.update(value) for value in [True, False, True, False, True]]
    assert outputs[-1] is True

