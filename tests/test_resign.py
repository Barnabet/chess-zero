from chesszero.resign import ResignGovernor


def gov(**kw):
    d = dict(arm_fp=0.05, disarm_fp=0.08, window=100, min_train_steps=10)
    d.update(kw)
    return ResignGovernor(**d)


def test_stays_disarmed_until_window_filled():
    g = gov()
    armed, fp, msg = g.update(fp=0, n=50, global_step=1000)
    assert (armed, fp, msg) == (False, None, None)   # only 50 of 100 triggers seen


def test_arms_below_threshold_and_reports():
    g = gov()
    g.update(fp=1, n=60, global_step=1000)
    armed, fp, msg = g.update(fp=1, n=60, global_step=1000)
    assert armed and fp is not None and fp < 0.05
    assert msg is not None and "armed" in msg.lower()


def test_min_train_steps_blocks_arming():
    g = gov()
    g.update(fp=0, n=60, global_step=5)
    armed, fp, msg = g.update(fp=0, n=60, global_step=5)
    assert not armed and fp is not None and msg is None


def test_hysteresis_holds_then_disarms():
    g = gov()
    g.update(fp=1, n=60, global_step=1000)
    g.update(fp=1, n=60, global_step=1000)          # armed (~1.7% FP)
    armed, fp, _ = g.update(fp=8, n=60, global_step=1000)   # window FP ~6% — hold
    assert armed
    armed, fp, msg = g.update(fp=30, n=60, global_step=1000)  # >8% — disarm
    assert not armed and msg is not None and "disarmed" in msg.lower()


def test_window_trims_old_generations():
    g = gov(window=100)
    g.update(fp=50, n=60, global_step=1000)   # terrible old gen
    g.update(fp=0, n=60, global_step=1000)
    armed, fp, _ = g.update(fp=0, n=60, global_step=1000)
    # window keeps the minimal trailing suffix holding >= 100 triggers:
    # the 50/60 gen fell out -> recent FP is low -> armed
    assert fp < 0.05 and armed
