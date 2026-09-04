# Tests for subpopulation-specific parameter input files --
#   parameters such as `IP_to_ISH_prop` and `ISH_to_HD_prop` that are
#   A x R within a subpopulation, but that the user wants to vary
#   across the L subpopulations of a metapopulation model

import clt_toolkit as clt
import flu_core as flu

import json
import numpy as np
import pytest

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"

common_params_filepath = base_path / "caseB_subpop_params.json"
subpop_specific_filepath = base_path / "caseB_subpop_specific_params.json"

subpop_names = ["subpop1", "subpop2"]


@pytest.fixture
def common_params():
    return clt.make_dataclass_from_json(common_params_filepath, flu.FluSubpopParams)


@pytest.fixture
def common_params_age_risk(common_params):
    """
    Common parameters whose `IP_to_ISH_prop` is a full A x R array, as
    in the real input files -- an A x R common value is what marks a
    parameter as age-risk, so this is what lets a scalar override be
    recognized and expanded.
    """

    return clt.updated_dataclass(
        common_params,
        {"IP_to_ISH_prop": np.full((common_params.num_age_groups,
                                    common_params.num_risk_groups), 0.01)})


@pytest.fixture
def write_subpop_specific_file(tmp_path):
    """Write a subpopulation-specific params dict to a temporary JSON file."""

    def _write(d: dict):
        filepath = tmp_path / "subpop_specific_params.json"
        with open(filepath, "w") as file:
            json.dump(d, file)
        return filepath

    return _write


def build_metapop_model_from_file(make_flu_metapop_model, filepath):
    """
    Build a metapop model and apply a subpopulation-specific params
    file to it, returning the model.
    """

    model = make_flu_metapop_model("binom_deterministic_no_round")

    updates_by_subpop = clt.load_subpop_specific_params_json(filepath,
                                                             subpop_names,
                                                             flu.FluSubpopParams,
                                                             model._subpop_models_ordered[0].params)

    for name, updates in updates_by_subpop.items():
        model.modify_subpop_params(name, updates)

    return model


def test_params_vary_by_subpop(common_params):
    """
    Each subpopulation gets its own A x R values, and the common
    parameters instance is not modified.
    """

    params_by_subpop = clt.make_subpop_params_dict(common_params_filepath,
                                                   subpop_names,
                                                   flu.FluSubpopParams,
                                                   subpop_specific_filepath)

    A = common_params.num_age_groups
    R = common_params.num_risk_groups

    for name in subpop_names:
        assert params_by_subpop[name].IP_to_ISH_prop.shape == (A, R)
        assert params_by_subpop[name].ISH_to_HD_prop.shape == (A, R)

    # caseB_subpop_specific_params.json gives subpop2 exactly double subpop1
    assert np.allclose(params_by_subpop["subpop2"].IP_to_ISH_prop,
                       2 * params_by_subpop["subpop1"].IP_to_ISH_prop)
    assert np.allclose(params_by_subpop["subpop2"].ISH_to_HD_prop,
                       2 * params_by_subpop["subpop1"].ISH_to_HD_prop)
    assert params_by_subpop["subpop1"].beta_baseline != \
           params_by_subpop["subpop2"].beta_baseline

    # Parameters absent from the subpopulation-specific file are shared
    assert params_by_subpop["subpop1"].E_to_I_rate == common_params.E_to_I_rate
    assert params_by_subpop["subpop2"].E_to_I_rate == common_params.E_to_I_rate

    # The common parameters must be untouched
    assert common_params.IP_to_ISH_prop == 0.01


def test_no_subpop_specific_file_gives_common_params(common_params):
    """
    Without a subpopulation-specific file, every subpopulation gets
    the common parameters.
    """

    params_by_subpop = clt.make_subpop_params_dict(common_params, subpop_names)

    for name in subpop_names:
        assert params_by_subpop[name] == common_params


def test_value_not_keyed_by_subpop_is_shared(common_params, write_subpop_specific_file):
    """
    A parameter given as a plain value (rather than a dictionary keyed
    by subpopulation name) applies to every subpopulation.
    """

    filepath = write_subpop_specific_file({"beta_baseline": 3.3})

    params_by_subpop = clt.make_subpop_params_dict(common_params,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    for name in subpop_names:
        assert params_by_subpop[name].beta_baseline == 3.3


def test_partial_override_falls_back_to_common(common_params_age_risk,
                                               write_subpop_specific_file):
    """
    A subpopulation omitted from a parameter's dictionary keeps the
    common value for that parameter -- expanded to A x R along with the
    overriding subpopulation's, so that the two stack.
    """

    A = common_params_age_risk.num_age_groups
    R = common_params_age_risk.num_risk_groups

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop1": 0.05}})

    params_by_subpop = clt.make_subpop_params_dict(common_params_age_risk,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    assert params_by_subpop["subpop1"].IP_to_ISH_prop.shape == (A, R)
    assert np.allclose(params_by_subpop["subpop1"].IP_to_ISH_prop, 0.05)

    assert params_by_subpop["subpop2"].IP_to_ISH_prop.shape == (A, R)
    assert np.allclose(params_by_subpop["subpop2"].IP_to_ISH_prop,
                       common_params_age_risk.IP_to_ISH_prop)


def test_scalar_parameter_stays_scalar(common_params, write_subpop_specific_file):
    """
    A parameter that is a scalar in the common parameters and is
    overridden with scalars is left scalar -- there is nothing marking
    it as age-risk (it could just as well be `beta_baseline`), and
    scalars stack across subpopulations without expansion anyway.
    """

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop1": 0.05,
                                                              "subpop2": 0.07}})

    params_by_subpop = clt.make_subpop_params_dict(common_params,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    assert params_by_subpop["subpop1"].IP_to_ISH_prop == 0.05
    assert params_by_subpop["subpop2"].IP_to_ISH_prop == 0.07


# ---------------------------------------------------------------------------
# Dimensions of age-risk values
# ---------------------------------------------------------------------------

# Every way of writing the same A x R value, for A = 5 and R = 1
# (`caseB_subpop_params.json`) -- scalar, length-A, A x 1, and A x R
AGE_RISK_SPELLINGS_A5_R1 = {
    "scalar": 0.03,
    "length_A": [0.03] * 5,
    "A_by_1": [[0.03]] * 5,
    "A_by_R": [[0.03]] * 5,
}


@pytest.mark.parametrize("spelling", sorted(AGE_RISK_SPELLINGS_A5_R1))
def test_age_risk_values_expanded(common_params_age_risk,
                                  write_subpop_specific_file,
                                  spelling):
    """
    Age-risk parameters may be given as a scalar, a length-A list, an
    A x 1 nested list, or an A x R nested list -- all are expanded to
    A x R, and all give the same result.
    """

    value = AGE_RISK_SPELLINGS_A5_R1[spelling]

    A = common_params_age_risk.num_age_groups
    R = common_params_age_risk.num_risk_groups

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop1": value}})

    params_by_subpop = clt.make_subpop_params_dict(common_params_age_risk,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    result = params_by_subpop["subpop1"].IP_to_ISH_prop

    # The expansion must actually happen -- `np.allclose` alone would
    #   broadcast a scalar and pass without it
    assert isinstance(result, np.ndarray)
    assert result.shape == (A, R)
    assert np.allclose(result, 0.03)


def test_A_by_1_broadcasts_across_risk_groups(write_subpop_specific_file):
    """
    With more than one risk group, an A x 1 nested list means "the same
    value for every risk group" -- the same as a length-A list.
    """

    common_params = clt.make_dataclass_from_json(common_params_filepath,
                                                 flu.FluSubpopParams)
    common_params = clt.updated_dataclass(
        common_params,
        {"num_risk_groups": 2,
         "IP_to_ISH_prop": np.full((common_params.num_age_groups, 2), 0.01)})

    A = common_params.num_age_groups
    age_only = [0.01, 0.02, 0.03, 0.04, 0.05]

    filepath = write_subpop_specific_file(
        {"IP_to_ISH_prop": {"subpop1": [[val] for val in age_only],
                            "subpop2": age_only}})

    params_by_subpop = clt.make_subpop_params_dict(common_params,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    expected = np.tile(np.asarray(age_only)[:, None], (1, 2))

    for name in subpop_names:
        assert params_by_subpop[name].IP_to_ISH_prop.shape == (A, 2)
        assert np.allclose(params_by_subpop[name].IP_to_ISH_prop, expected)


def test_mixed_dimensions_across_subpops_stack_in_torch(make_flu_metapop_model,
                                                        write_subpop_specific_file):
    """
    Subpopulations may spell the same age-risk parameter differently
    (one a scalar, one an A x R list), and the values must still stack
    into a rectangular L x A x R tensor.

    Regression test: a scalar left un-expanded alongside an array made
    `update_params_tensors` build a ragged array, which `numpy` rejects.
    """

    filepath = write_subpop_specific_file(
        {"IP_to_ISH_prop": {"subpop1": [[0.1], [0.1], [0.1], [0.1], [0.1]],
                            "subpop2": 0.2}})

    model = build_metapop_model_from_file(make_flu_metapop_model, filepath)

    params_tensors = model.get_flu_torch_inputs()["params_tensors"]

    assert tuple(params_tensors.IP_to_ISH_prop.shape) == (2, 5, 1)
    assert np.allclose(np.asarray(params_tensors.IP_to_ISH_prop[0]), 0.1)
    assert np.allclose(np.asarray(params_tensors.IP_to_ISH_prop[1]), 0.2)


def test_partial_override_stacks_in_torch(make_flu_metapop_model,
                                          write_subpop_specific_file):
    """
    Overriding an age-risk parameter for only some subpopulations must
    also produce a rectangular L x A x R tensor -- the subpopulations
    left out fall back to the (possibly scalar) common value, which is
    expanded along with the rest.
    """

    filepath = write_subpop_specific_file(
        {"IP_to_ISH_prop": {"subpop1": [[0.1], [0.1], [0.1], [0.1], [0.1]]}})

    model = build_metapop_model_from_file(make_flu_metapop_model, filepath)

    params_tensors = model.get_flu_torch_inputs()["params_tensors"]

    common_value = model._subpop_models_ordered[1].params.IP_to_ISH_prop

    assert tuple(params_tensors.IP_to_ISH_prop.shape) == (2, 5, 1)
    assert np.allclose(np.asarray(params_tensors.IP_to_ISH_prop[0]), 0.1)
    assert np.allclose(np.asarray(params_tensors.IP_to_ISH_prop[1]), common_value)


def test_torch_params_stack_by_subpop(make_flu_metapop_model):
    """
    Per-subpopulation A x R values become the L x A x R tensors that
    the torch model runs on, in subpopulation order.
    """

    model = build_metapop_model_from_file(make_flu_metapop_model,
                                          subpop_specific_filepath)

    params_tensors = model.get_flu_torch_inputs()["params_tensors"]

    for idx, subpop_model in enumerate(model._subpop_models_ordered.values()):
        assert np.allclose(np.asarray(params_tensors.IP_to_ISH_prop[idx]),
                           subpop_model.params.IP_to_ISH_prop)
        assert np.allclose(np.asarray(params_tensors.ISH_to_HD_prop[idx]),
                           subpop_model.params.ISH_to_HD_prop)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_unknown_subpop_name_raises(common_params, write_subpop_specific_file):
    """
    A subpopulation name that is not in the model raises, so that
    typos are not silently ignored.
    """

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop3": 0.05}})

    with pytest.raises(ValueError, match="unknown subpopulation"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


def test_unknown_param_name_raises(common_params, write_subpop_specific_file):
    """
    A parameter name that is not a field of the parameters dataclass
    raises, so that typos are not silently ignored.
    """

    filepath = write_subpop_specific_file({"IP_to_ISH_props": {"subpop1": 0.05}})

    with pytest.raises(ValueError, match="unknown parameter"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


@pytest.mark.parametrize("param_name", clt.SUBPOP_INVARIANT_PARAMS)
def test_structural_param_raises(common_params, write_subpop_specific_file, param_name):
    """
    A and R define the grid that subpopulation parameters are stacked
    on, so they cannot vary by subpopulation.
    """

    filepath = write_subpop_specific_file({param_name: {"subpop1": 3}})

    with pytest.raises(ValueError, match="cannot vary by subpopulation"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


def test_wrong_age_risk_shape_raises(common_params, write_subpop_specific_file):
    """
    An age-risk value whose length does not match the number of age
    groups raises.
    """

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop1": [0.01, 0.02, 0.03]}})

    with pytest.raises(ValueError, match="Age-risk parameters must be"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


def test_wrong_matrix_shape_raises(common_params, write_subpop_specific_file):
    """
    A non-age-risk array value must match the shape of its common value.
    """

    filepath = write_subpop_specific_file({"total_contact_matrix": {"subpop1": [[1, 2], [3, 4]]}})

    with pytest.raises(ValueError, match="matching the common value"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


def test_null_value_raises(common_params, write_subpop_specific_file):
    """
    A `null` value is a mistake -- a subpopulation is left on the
    common value by being omitted, not by being set to null.
    """

    filepath = write_subpop_specific_file({"IP_to_ISH_prop": {"subpop1": None}})

    with pytest.raises(ValueError, match="got null"):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=filepath)


def test_missing_file_raises(common_params, tmp_path):
    """
    A mistyped filepath raises rather than being silently ignored --
    callers for whom the file is genuinely optional check for it first.
    """

    with pytest.raises(FileNotFoundError):
        clt.make_subpop_params_dict(common_params,
                                    subpop_names,
                                    subpop_specific_filepath=tmp_path / "nope.json")


def test_dataclass_ref_required_with_filepath():
    """
    Building from a filepath needs the dataclass to build into.
    """

    with pytest.raises(ValueError, match="dataclass_ref is required"):
        clt.make_subpop_params_dict(common_params_filepath, subpop_names)


# ---------------------------------------------------------------------------
# Optional validation arguments
# ---------------------------------------------------------------------------

def test_without_dataclass_ref_param_names_are_not_checked(common_params,
                                                           write_subpop_specific_file):
    """
    `dataclass_ref` is what parameter names are validated against --
    without it, `load_subpop_specific_params_json` accepts any name
    (the caller is then responsible for what it does with them).
    """

    filepath = write_subpop_specific_file({"not_a_real_param": {"subpop1": 0.05}})

    updates_by_subpop = clt.load_subpop_specific_params_json(filepath,
                                                             subpop_names,
                                                             common_params=common_params)

    assert updates_by_subpop["subpop1"]["not_a_real_param"] == 0.05


def test_without_common_params_values_are_stored_as_written(write_subpop_specific_file):
    """
    Recognizing a parameter as age-risk needs the common parameters
    (they are where A and R come from) -- without them, values are
    stored exactly as written and no shape checking happens.
    """

    filepath = write_subpop_specific_file(
        {"IP_to_ISH_prop": {"subpop1": [0.01, 0.02, 0.03]}})

    updates_by_subpop = clt.load_subpop_specific_params_json(filepath,
                                                             subpop_names,
                                                             flu.FluSubpopParams)

    assert np.allclose(updates_by_subpop["subpop1"]["IP_to_ISH_prop"],
                       [0.01, 0.02, 0.03])
    assert "subpop2" not in updates_by_subpop


def test_comment_keys_are_ignored(common_params, write_subpop_specific_file):
    """
    `JSON` has no comment syntax, so underscore-prefixed keys are
    ignored in both the common and the subpopulation-specific files.
    """

    filepath = write_subpop_specific_file({"_comment": "not a parameter",
                                           "beta_baseline": {"subpop1": 4.4}})

    params_by_subpop = clt.make_subpop_params_dict(common_params,
                                                   subpop_names,
                                                   subpop_specific_filepath=filepath)

    assert params_by_subpop["subpop1"].beta_baseline == 4.4

    # And in the common parameters file, where an unknown key would
    #   otherwise be an unexpected keyword argument
    with_comment = dict(json.load(open(common_params_filepath)),
                        _comment="not a parameter")
    from_comment = clt.make_dataclass_from_dict(flu.FluSubpopParams, with_comment)

    # Compared field by field -- dataclass `__eq__` on array fields
    #   raises "truth value of an array is ambiguous"
    for field_name in flu.FluSubpopParams.__dataclass_fields__:
        assert np.all(np.equal(getattr(from_comment, field_name),
                               getattr(common_params, field_name)))
