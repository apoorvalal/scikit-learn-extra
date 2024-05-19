import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklearn_extra.preprocessing import BernsteinFeatures

from sklearn.utils._testing import (
    assert_array_almost_equal,
)

feature_1d = np.array([0, 0.5, 1, np.nan]).reshape(-1, 1)
feature_2d = np.array([[0, 0.25], [0.5, 0.5], [np.nan, 0.75]])


@parametrize_with_checks([BernsteinFeatures()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)


def test_correct_param_types():
    with pytest.raises(ValueError):
        BernsteinFeatures(degree="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(bias="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(na_value="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(interactions="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1.5).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(bias=1.5).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(na_value="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(interactions=1.5).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=-1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(bias=-1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(na_value=-1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(interactions=-1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1.5).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1, na_value=1).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1, na_value=1.5).fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1, na_value=1, interactions=1).fit(
            feature_1d
        )

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=1, bias=1, na_value=1, interactions=1.5).fit(
            feature_1d
        )

    with pytest.raises(ValueError):
        BernsteinFeatures(
            degree=1, bias=1, na_value=1, interactions=1, unknown=1
        ).fit(feature_1d)


def test_correct_output_one_feature():
    bbt = BernsteinFeatures(degree=2).fit(np.empty(0))
    output = bbt.transform(feature_1d)
    expected_output = np.array(
        [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0], [0.0, 0.0]]
    )
    assert_array_almost_equal(output, expected_output)


def test_correct_output_two_features():
    bbt = BernsteinFeatures(degree=2).fit(np.empty(0))
    output = bbt.transform(feature_2d)
    expected_output = np.array(
        [
            [0.0, 0.0, 0.25, 0.0625],
            [0.5, 0.25, 0.5, 0.25],
            [0.0, 0.0, 0.75, 0.5625],
        ]
    )
    assert_array_almost_equal(output, expected_output)


def test_correct_output_interactions():
    bbt = BernsteinFeatures(degree=2, interactions=True).fit(np.empty(0))
    output = bbt.transform(feature_2d)
    expected_output = np.array(
        [
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0625, 0.0, 0.0],
            [0.5, 0.25, 0.5, 0.25, 0.125, 0.25, 0.125, 0.0625],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_array_almost_equal(output, expected_output)
