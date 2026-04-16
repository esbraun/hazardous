"""Tests for custom classifier support in SurvivalBoost."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from hazardous import SurvivalBoost
from hazardous._ipcw import _build_warm_start_estimator, _step_warm_start_fit
from hazardous.data import make_synthetic_competing_weibull


@pytest.fixture(scope="module")
def train_test_data():
    X, y = make_synthetic_competing_weibull(return_X_y=True, random_state=0)
    return train_test_split(X, y, random_state=0)


# ---------------------------------------------------------------------------
# _build_warm_start_estimator unit tests
# ---------------------------------------------------------------------------


def test_build_warm_start_estimator_default():
    """Default (estimator=None) builds an HGB with warm_start and max_iter=1."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = _build_warm_start_estimator(
        estimator=None,
        estimator_params=None,
        default_params=dict(learning_rate=0.05, max_leaf_nodes=31),
    )
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert clf.warm_start is True
    assert clf.max_iter == 1
    assert clf.learning_rate == 0.05


def test_build_warm_start_estimator_custom_hgb_params():
    """estimator_params overrides default_params when estimator=None."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = _build_warm_start_estimator(
        estimator=None,
        estimator_params={"learning_rate": 0.2, "max_leaf_nodes": 10},
        default_params=dict(learning_rate=0.05, max_leaf_nodes=31),
    )
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert clf.learning_rate == 0.2
    assert clf.max_leaf_nodes == 10
    # Internal loop params are always enforced
    assert clf.warm_start is True
    assert clf.max_iter == 1


def test_build_warm_start_estimator_custom_class_max_iter():
    """Custom estimator with max_iter gets it forced to 1."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = _build_warm_start_estimator(
        estimator=HistGradientBoostingClassifier,
        estimator_params={"learning_rate": 0.1},
        default_params=None,
    )
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert clf.warm_start is True
    assert clf.max_iter == 1


def test_build_warm_start_estimator_custom_class_n_estimators():
    """Custom estimator with n_estimators gets it forced to 1."""
    clf = _build_warm_start_estimator(
        estimator=RandomForestClassifier,
        estimator_params={"n_jobs": 1},
        default_params=None,
    )
    assert isinstance(clf, RandomForestClassifier)
    assert clf.warm_start is True
    assert clf.n_estimators == 1


def test_build_warm_start_estimator_no_iteration_attr():
    """Estimators with neither max_iter nor n_estimators raise ValueError."""

    class _NoIterAttrEstimator:
        """Minimal sklearn-like classifier without max_iter or n_estimators."""

        def __init__(self, warm_start=True):
            self.warm_start = warm_start

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.ones((len(X), 2)) / 2

    with pytest.raises(ValueError, match="must expose either 'max_iter' or"):
        _build_warm_start_estimator(
            estimator=_NoIterAttrEstimator,
            estimator_params=None,
            default_params=None,
        )


# ---------------------------------------------------------------------------
# _step_warm_start_fit unit tests
# ---------------------------------------------------------------------------


def test_step_warm_start_fit_max_iter():
    """_step_warm_start_fit increments max_iter for HGB-style estimators."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.integers(0, 2, size=50)

    clf = HistGradientBoostingClassifier(warm_start=True, max_iter=1)
    _step_warm_start_fit(clf, X, y)
    assert clf.max_iter == 2
    _step_warm_start_fit(clf, X, y)
    assert clf.max_iter == 3


def test_step_warm_start_fit_n_estimators():
    """_step_warm_start_fit increments n_estimators for RF-style estimators."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.integers(0, 2, size=50)

    clf = RandomForestClassifier(warm_start=True, n_estimators=1, random_state=0)
    _step_warm_start_fit(clf, X, y)
    assert clf.n_estimators == 2
    _step_warm_start_fit(clf, X, y)
    assert clf.n_estimators == 3


# ---------------------------------------------------------------------------
# SurvivalBoost integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_default_still_works(train_test_data, ipcw_strategy):
    """Existing default behaviour is unchanged when no estimator is provided."""
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        random_state=0,
    )
    est.fit(X_train, y_train)
    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_custom_hgb_params(train_test_data, ipcw_strategy):
    """estimator_params overrides default HGB hyperparameters."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator_params={"learning_rate": 0.1, "max_leaf_nodes": 15},
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, HistGradientBoostingClassifier)
    assert est.estimator_.learning_rate == 0.1
    assert est.estimator_.max_leaf_nodes == 15

    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_gradient_boosting_classifier(train_test_data, ipcw_strategy):
    """sklearn GradientBoostingClassifier (n_estimators) works as custom estimator."""
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator=GradientBoostingClassifier,
        estimator_params={"learning_rate": 0.1},
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, GradientBoostingClassifier)
    assert est.estimator_.warm_start is True
    # starts at 1, incremented once per boosting step: 1 + n_iter = 4
    assert est.estimator_.n_estimators == 1 + est.n_iter

    cif = est.predict_cumulative_incidence(X_test)
    assert np.all(cif >= 0)
    assert np.all(cif <= 1)
    assert_allclose(cif.sum(axis=1), 1.0)


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_random_forest(train_test_data, ipcw_strategy):
    """sklearn RandomForestClassifier (n_estimators) works as custom estimator."""
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator=RandomForestClassifier,
        estimator_params={"random_state": 0, "n_jobs": 1},
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, RandomForestClassifier)
    assert est.estimator_.warm_start is True

    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)


# ---------------------------------------------------------------------------
# Optional third-party library tests (skipped when not installed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_xgboost(train_test_data, ipcw_strategy):
    xgb = pytest.importorskip("xgboost")
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator=xgb.XGBClassifier,
        estimator_params={
            "learning_rate": 0.1,
            "max_depth": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 0,
        },
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, xgb.XGBClassifier)
    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_lightgbm(train_test_data, ipcw_strategy):
    lgbm = pytest.importorskip("lightgbm")
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator=lgbm.LGBMClassifier,
        estimator_params={"learning_rate": 0.1, "max_depth": 3, "random_state": 0},
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, lgbm.LGBMClassifier)
    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)


@pytest.mark.parametrize("ipcw_strategy", ["kaplan-meier", "alternating"])
def test_survival_boost_catboost(train_test_data, ipcw_strategy):
    catboost = pytest.importorskip("catboost")
    X_train, X_test, y_train, _ = train_test_data
    est = SurvivalBoost(
        n_iter=3,
        show_progressbar=False,
        ipcw_strategy=ipcw_strategy,
        estimator=catboost.CatBoostClassifier,
        estimator_params={"learning_rate": 0.1, "depth": 3, "verbose": 0},
        random_state=0,
    )
    est.fit(X_train, y_train)
    assert isinstance(est.estimator_, catboost.CatBoostClassifier)
    survival = est.predict_survival_function(X_test)
    assert np.all(survival >= 0)
    assert np.all(survival <= 1)
