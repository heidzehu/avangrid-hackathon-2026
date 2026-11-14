import pandas as pd
import pytest
from avangrid.integration import integrate, run


def fixtures():
    cls = pd.DataFrame({"customer": [1, 2], "f_ev": [True, False], "f_heat_pump": [False, True],
                        "f_hp_candidate": [False, False], "f_solar_pv": [True, False]})
    seg = pd.DataFrame({"customer": [1, 2], "segment": ["Evening peak", "Winter peak"], "cluster": [0, 1]})
    sizing = pd.DataFrame({"customer_number": [1, 1, 2, 2], "option": ["baseline", "dr", "baseline", "battery_1unit"],
                           "recommended": [False, True, True, False], "npv_usd": [0, 10, 0, -5],
                           "critical_hour_reduction_kw": [0, 1.5, 0, 0.9]})
    return cls, seg, sizing


def test_join_keeps_recommendations_and_summarizes():
    cls, seg, sizing = fixtures()
    out, summary, audit = integrate(cls, seg, sizing)
    assert len(out) == 2
    assert set(out.option) == {"dr", "baseline"}
    assert audit["fully_matched"] == 2
    assert out.loc[out.customer == 1, "npv_usd"].iloc[0] == 10
    assert summary.customers.sum() == 2
    assert summary.sum_customer_critical_reduction_kw.sum() == 1.5


def test_missing_labels_are_visible_not_imputed():
    cls, seg, sizing = fixtures()
    out, summary, audit = integrate(cls.iloc[:1], seg.iloc[:1], sizing)
    assert audit["fully_matched"] == 1
    assert audit["unmatched_customer_ids"] == [2]
    assert "Unmatched / not classified" in set(summary.segment)
    assert not out.loc[out.customer == 2, "classification_available"].iloc[0]


def test_duplicate_ids_rejected():
    cls, seg, sizing = fixtures()
    with pytest.raises(ValueError, match="duplicate"):
        integrate(pd.concat([cls, cls.iloc[:1]]), seg, sizing)


def test_duplicate_optimized_recommendations_rejected():
    cls, seg, sizing = fixtures()
    with pytest.raises(ValueError, match="duplicate"):
        integrate(cls, seg, pd.concat([sizing, sizing.iloc[[1]]]))


def test_malformed_inputs_rejected():
    cls, seg, sizing = fixtures()
    with pytest.raises(ValueError, match="missing"):
        integrate(cls.drop(columns=["f_ev"]), seg, sizing)


def test_require_complete_prevents_writing(tmp_path, monkeypatch):
    cls, seg, sizing = fixtures()
    p1, p2, p3 = tmp_path / "c.parquet", tmp_path / "s.parquet", tmp_path / "z.csv"
    p1.touch(); p2.touch()
    sizing.to_csv(p3, index=False)
    def parquet(path):
        return cls.iloc[:1] if path == p1 else seg
    monkeypatch.setattr(pd, "read_parquet", parquet)
    with pytest.raises(ValueError, match="matched"):
        run(p1, p2, p3, tmp_path / "out", require_complete=True)
    assert not (tmp_path / "out" / "part3_integrated_segment_summary.csv").exists()


def test_full_matrix_alignment_guard(tmp_path, monkeypatch):
    import numpy as np
    cls, seg, sizing = fixtures()
    c, s, z, m = (tmp_path / x for x in ("c.parquet", "s.parquet", "size.csv", "customers.npy"))
    c.touch(); s.touch(); sizing.to_csv(z, index=False)
    np.save(m, np.array([1, 2, 3]))
    monkeypatch.setattr(pd, "read_parquet", lambda path: cls if path == c else seg)
    with pytest.raises(ValueError, match="does not match"):
        run(c, s, z, tmp_path / "out", matrix_customers=m)
