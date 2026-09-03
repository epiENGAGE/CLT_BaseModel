import numpy as np
import json
from pathlib import Path
from typing import ClassVar, Optional, Sequence, Type, Union
from typing import Protocol

from .utils import to_AR_array, updated_dataclass

# JSON filepaths may be given as strings or `Path` objects
FilePath = Union[str, Path]


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[dict]


def convert_dict_vals_lists_to_arrays(d: dict) -> dict:
    """
    Converts dictionary of lists to dictionary of arrays
    to support `numpy` operations.
    """

    for key, val in d.items():
        if type(val) is list:
            d[key] = np.asarray(val)

    return d


def load_json_new_dict(json_filepath: FilePath) -> dict:
    """
    Loads specified `JSON` file into new dictionary.
    Lists are automatically converted to numpy arrays for
    computational compatibility, since `JSON` does not natively
    support `np.ndarray`.

    Args:
        json_filepath (str):
            Full `JSON` filepath.

    Returns:
        (dict):
            Dictionary loaded with `JSON` information.
    """

    # Note: the "with open" is important for file handling
    #   and avoiding resource leaks -- otherwise,
    #   we have to manually close the file, which is a bit
    #   more cumbersome
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # json does not support numpy, so we must convert
    #   lists to numpy arrays
    return convert_dict_vals_lists_to_arrays(data)


def load_json_augment_dict(json_filepath: FilePath,
                           d: dict) -> dict:
    """
    Augments pre-existing dictionary with information
    from `JSON` file -- if keys already exist, the previous values
    are overriden, otherwise the new key-value pairs are added.
    Lists are automatically converted to numpy arrays for
    computational compatibility, since `JSON` does not natively
    support `np.ndarray`.

    Args:
        json_filepath (str):
            Full `JSON` filepath.
        d (dict):
            Dictionary to be augmented with new `JSON` values.

    Returns:
        (dict):
            Dictionary loaded with `JSON` information.
    """

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    data = convert_dict_vals_lists_to_arrays(data)

    for key, val in data.items():
        d[key] = val

    return d


def make_dataclass_from_dict(dataclass_ref: Type[DataClassProtocol],
                             d: dict) -> DataClassProtocol:
    """
    Create instance of class dataclass_ref,
    based on information in dictionary.

    Args:
        dataclass_ref (Type[DataClassProtocol]):
            (class, not instance) from which to create instance --
            must have dataclass decorator.
        d (dict):
            all keys and values respectively must match name and datatype
            of dataclass_ref instance attributes -- except for keys
            beginning with an underscore, which are ignored, so that a
            `JSON` file can carry comments (`JSON` has no comment syntax).

    Returns:
        DataClassProtocol:
            instance of dataclass_ref with attributes dynamically
            assigned by json_filepath file contents.
    """

    d = convert_dict_vals_lists_to_arrays(d)

    d = {key: val for key, val in d.items() if not key.startswith("_")}

    return dataclass_ref(**d)


def make_dataclass_from_json(json_filepath: FilePath,
                             dataclass_ref: Type[DataClassProtocol]) -> DataClassProtocol:
    """
    Create instance of class dataclass_ref,
    based on information in json_filepath.

    Args:
        json_filepath (str):
            path to json file (path includes actual filename
            with suffix ".json") -- all json fields must
            match name and datatype of dataclass_ref instance
            attributes.
        dataclass_ref (Type[DataClassProtocol]):
            (class, not instance) from which to create instance --
            must have dataclass decorator.

    Returns:
        DataClassProtocol:
            instance of dataclass_ref with attributes dynamically
            assigned by json_filepath file contents.
    """

    d = load_json_new_dict(json_filepath)

    return make_dataclass_from_dict(dataclass_ref, d)


# Parameters that describe the *structure* of a subpopulation's age-risk
#   grid rather than its epidemiology. A metapopulation model combines
#   its subpopulations into (L, A, R) tensors, so A and R must be
#   identical across subpopulations -- letting these vary would fail
#   later with an obscure shape error
SUBPOP_INVARIANT_PARAMS = ("num_age_groups", "num_risk_groups")


def _normalize_subpop_override(param_name: str,
                               subpop_name: str,
                               value,
                               common_value,
                               num_age_groups: Optional[int],
                               num_risk_groups: Optional[int]) -> tuple:
    """
    Validate (and when possible normalize) one subpopulation-specific
    parameter value against the corresponding common (shared) value.

    Age-risk parameters may be given as a scalar, as a
    length-`num_age_groups` list, as a `num_age_groups` x 1 nested
    list, or as a full `num_age_groups` x `num_risk_groups` nested
    list -- all four are expanded to `num_age_groups` x
    `num_risk_groups` (the first three by broadcasting across risk
    groups). A parameter counts as age-risk when its common value is
    an A x R array, or when its common value is a scalar (or absent)
    and the subpopulation-specific value is itself shaped like A,
    A x 1, or A x R. Every other parameter must match the shape of its
    common value exactly.

    Args:
        param_name (str):
            name of the parameter being overridden -- used for
            error messages.
        subpop_name (str):
            name of the subpopulation whose value is being checked --
            used for error messages.
        value:
            subpopulation-specific value, as read from `JSON`.
        common_value:
            value of the same parameter in the common (shared)
            parameters, or `None` if the common parameters do not
            specify it.
        num_age_groups (Optional[int]):
            number of age groups (A), or `None` if unknown.
        num_risk_groups (Optional[int]):
            number of risk groups (R), or `None` if unknown.

    Returns:
        (tuple):
            2-tuple `(normalized_value, is_age_risk)` -- for an
            age-risk parameter, `normalized_value` is an A x R array
            and `is_age_risk` is `True`; otherwise `normalized_value`
            is `value` unchanged and `is_age_risk` is `False`.
    """

    where = f"parameter {param_name!r}, subpopulation {subpop_name!r}"

    if value is None:
        raise ValueError(
            f"Invalid subpopulation-specific value for {where}: got null. "
            "To leave a subpopulation on the common value, omit it entirely "
            "rather than giving it null."
        )

    # `JSON` has no array type, so nested lists arrive as lists --
    #   convert so that the model can do numpy operations on them
    value_array = np.asarray(value)
    common_array = None if common_value is None else np.asarray(common_value)

    if num_age_groups is not None and num_risk_groups is not None:
        age_risk_shape = (num_age_groups, num_risk_groups)

        # The common value pins down the shape when it is itself A x R,
        #   and so does a scalar common value that is given an array
        #   here -- an array replacing a scalar can only be an age-risk
        #   expansion of it. When the parameter is missing from the
        #   common parameters entirely there is nothing to infer from,
        #   so only expand values that already look age-risk shaped
        #   (a per-subpopulation contact matrix, say, must be left alone).
        common_is_age_risk = common_array is not None and common_array.shape == age_risk_shape
        common_is_scalar = common_array is not None and common_array.ndim == 0
        value_is_age_risk = value_array.shape in ((num_age_groups,),
                                                  (num_age_groups, 1),
                                                  age_risk_shape)

        if common_is_age_risk or \
                (common_is_scalar and value_array.ndim > 0) or \
                (common_array is None and value_is_age_risk):
            return _to_age_risk_override(where,
                                         value_array,
                                         num_age_groups,
                                         num_risk_groups), True

        # A scalar override of a scalar common value is still an
        #   age-risk parameter whenever the model's A x R grid is
        #   known and some *other* subpopulation gives it as an array;
        #   the caller resolves that (see `_expand_age_risk_fallbacks`),
        #   so report the scalar as-is here.

    if common_array is not None and common_array.ndim > 0 and \
            value_array.shape != common_array.shape:
        raise ValueError(
            f"Invalid subpopulation-specific value for {where}: expected shape "
            f"{common_array.shape} (matching the common value), got {value_array.shape}."
        )

    # Scalars stay scalars -- np.asarray would turn them into
    #   0-dimensional arrays, which is not what the rest of the code expects
    return (value if value_array.ndim == 0 else value_array), False


def _to_age_risk_override(where: str,
                          value,
                          num_age_groups: int,
                          num_risk_groups: int) -> np.ndarray:
    """
    Expand one age-risk value to an A x R array, re-raising shape
    failures with a message naming the offending parameter and
    subpopulation.

    Extends `to_AR_array` with the A x 1 case, which `to_AR_array`
    rejects: a single column is unambiguously "the same value for every
    risk group", and writing one value per row is the natural way to
    give an age-only parameter in a nested-list file.

    Args:
        where (str):
            description of the parameter and subpopulation being
            expanded -- used for error messages.
        value:
            scalar or array to expand.
        num_age_groups (int):
            number of age groups (A).
        num_risk_groups (int):
            number of risk groups (R).

    Returns:
        (np.ndarray of shape (A, R))
    """

    value_array = np.asarray(value)

    # `to_AR_array` accepts (A,) but not (A, 1) -- both mean the same
    #   thing here, so flatten the single column and let it broadcast
    if value_array.ndim == 2 and value_array.shape == (num_age_groups, 1) \
            and num_risk_groups != 1:
        value_array = value_array[:, 0]

    try:
        return to_AR_array(value_array, num_age_groups, num_risk_groups)
    except ValueError as error:
        raise ValueError(
            f"Invalid subpopulation-specific value for {where}: {error} "
            f"Age-risk parameters must be a scalar, a length-{num_age_groups} "
            f"list, a {num_age_groups} x 1 nested list, or a "
            f"{num_age_groups} x {num_risk_groups} nested list."
        ) from error


def _expand_age_risk_fallbacks(params_by_subpop: dict,
                               age_risk_param_names: set,
                               subpop_names: Sequence[str],
                               common_params: Optional[DataClassProtocol],
                               num_age_groups: int,
                               num_risk_groups: int) -> None:
    """
    Modify `params_by_subpop` in place so that every subpopulation has
    an A x R value for every age-risk parameter that any subpopulation
    overrides.

    Subpopulations left out of a parameter fall back to the common
    value, which may be a scalar; a subpopulation may also override an
    age-risk parameter with a scalar of its own. Either way the
    parameter would end up with different shapes in different
    subpopulations, and a metapopulation model stacks subpopulation
    parameters into an (L, A, R) tensor -- a ragged stack fails with an
    opaque `numpy` error. Expanding every subpopulation's value to
    A x R here keeps the stack rectangular.

    Args:
        params_by_subpop (dict):
            `{subpop name: {param name: value}}`, modified in place.
        age_risk_param_names (set):
            names of the parameters that were recognized as age-risk
            for at least one subpopulation.
        subpop_names (Sequence[str]):
            names of every subpopulation in the model.
        common_params (Optional[DataClassProtocol]):
            instance of the common (shared) parameters.
        num_age_groups (int):
            number of age groups (A).
        num_risk_groups (int):
            number of risk groups (R).
    """

    for param_name in age_risk_param_names:
        for subpop_name in subpop_names:

            updates = params_by_subpop.setdefault(subpop_name, {})
            where = f"parameter {param_name!r}, subpopulation {subpop_name!r}"

            if param_name in updates:
                value = updates[param_name]
            else:
                # This subpopulation keeps the common value -- expand it
                #   too, so that it stacks with the overridden ones
                value = getattr(common_params, param_name, None)
                if value is None:
                    raise ValueError(
                        f"Cannot fall back to the common value for {where}: the "
                        f"common parameters do not specify {param_name!r}. Give "
                        f"every subpopulation a value for it, or add it to the "
                        f"common parameters."
                    )

            updates[param_name] = _to_age_risk_override(where,
                                                        value,
                                                        num_age_groups,
                                                        num_risk_groups)


def load_subpop_specific_params_json(json_filepath: FilePath,
                                     subpop_names: Sequence[str],
                                     dataclass_ref: Optional[Type[DataClassProtocol]] = None,
                                     common_params: Optional[DataClassProtocol] = None) -> dict:
    """
    Load subpopulation-specific parameter values from a `JSON` file
    and reorganize them by subpopulation.

    The file is parameter-major -- each key is a parameter name, and
    its value is either a dictionary mapping subpopulation names to
    that subpopulation's value, or a single value shared by every
    subpopulation. For example, for 5 age groups and 2 risk groups::

        {
            "IP_to_ISH_prop": {
                "east": [[0.007, 0.054], [0.003, 0.009], ...],
                "west": [[0.005, 0.041], [0.002, 0.007], ...]
            },
            "ISH_to_HD_prop": {
                "east": [[0.013], [0.010], ...],
                "west": [[0.011], [0.008], ...]
            },
            "beta_baseline": {"east": 1.5, "west": 2.1}
        }

    Stacking the per-subpopulation A x R arrays of one parameter gives
    the L x A x R information described by such a file (where L is the
    number of subpopulations), but keying by name rather than by
    position means the file does not silently depend on subpopulation
    ordering.

    Age-risk values may be given as a full A x R nested list (as
    `IP_to_ISH_prop` is above), as an A x 1 nested list or a length-A
    list (as `ISH_to_HD_prop` is above -- both broadcast the same value
    across risk groups), or as a scalar. Once a parameter is recognized
    as age-risk -- because its common value is an A x R array, or
    because some subpopulation gives it as an array -- *every*
    subpopulation's value for it is expanded to A x R, so that a
    metapopulation model can stack them into a rectangular L x A x R
    tensor. A parameter that is scalar in the common parameters and
    scalar in every override stays scalar: nothing identifies it as
    age-risk, and scalars stack without expansion anyway. Recognizing
    age-risk parameters at all needs `common_params`; without it,
    values are stored as written.

    A subpopulation may be omitted from a parameter's dictionary -- it
    then keeps the common (shared) value for that parameter, expanded
    to A x R along with the rest if the parameter is age-risk. An
    unrecognized subpopulation name raises `ValueError`, so that typos
    are not silently ignored, and so does any of
    `SUBPOP_INVARIANT_PARAMS`, which must be the same in every
    subpopulation.

    Keys beginning with an underscore are ignored, so that a file can
    carry comments (`JSON` has no comment syntax).

    Args:
        json_filepath (str | Path):
            full `JSON` filepath.
        subpop_names (Sequence[str]):
            names of every subpopulation in the model -- names in the
            file must belong to this collection.
        dataclass_ref (Optional[Type[DataClassProtocol]]):
            parameters dataclass (class, not instance) that the values
            will be assigned to -- if given, parameter names in the file
            are checked against its fields.
        common_params (Optional[DataClassProtocol]):
            instance of the common (shared) parameters -- if given, it
            is used to identify age-risk parameters, validate the shape
            of each subpopulation-specific value, and fill in A x R
            values for subpopulations that do not override an age-risk
            parameter.

    Returns:
        (dict):
            dictionary mapping subpopulation name to a dictionary of
            that subpopulation's parameter values (suitable for passing
            to `updated_dataclass`). Every subpopulation that appears
            in the file has an entry; so does any subpopulation that
            needs an expanded fallback value for an age-risk parameter.
    """

    raw = load_json_new_dict(json_filepath)

    if dataclass_ref is not None:
        valid_param_names = set(dataclass_ref.__dataclass_fields__)
        dataclass_name = getattr(dataclass_ref, "__name__", "the parameters dataclass")
    else:
        valid_param_names = None
        dataclass_name = ""

    num_age_groups = getattr(common_params, "num_age_groups", None)
    num_risk_groups = getattr(common_params, "num_risk_groups", None)

    params_by_subpop = {}

    # Parameters recognized as age-risk for at least one subpopulation --
    #   every subpopulation's value for these is expanded to A x R below
    age_risk_param_names = set()

    for param_name, param_val in raw.items():

        # `JSON` has no comment syntax -- underscore-prefixed keys are
        #   the file's stand-in, and are not parameters
        if param_name.startswith("_"):
            continue

        if valid_param_names is not None and param_name not in valid_param_names:
            raise ValueError(
                f"{json_filepath} specifies unknown parameter {param_name!r} -- "
                f"it is not a field of {dataclass_name}."
            )

        if param_name in SUBPOP_INVARIANT_PARAMS:
            raise ValueError(
                f"{json_filepath} specifies parameter {param_name!r}, which cannot "
                f"vary by subpopulation -- a metapopulation model combines its "
                f"subpopulations into (L, A, R) tensors, so every subpopulation "
                f"must share the same age-risk grid. Set it in the common "
                f"parameters instead."
            )

        common_value = getattr(common_params, param_name, None)

        # A value that is not keyed by subpopulation name is shared
        #   by every subpopulation
        if isinstance(param_val, dict):
            val_by_subpop = param_val
        else:
            val_by_subpop = {name: param_val for name in subpop_names}

        unknown_names = set(val_by_subpop) - set(subpop_names)
        if unknown_names:
            raise ValueError(
                f"{json_filepath} specifies parameter {param_name!r} for unknown "
                f"subpopulation(s) {sorted(unknown_names)} -- valid names are "
                f"{sorted(subpop_names)}."
            )

        for subpop_name, subpop_val in val_by_subpop.items():
            value, is_age_risk = _normalize_subpop_override(param_name,
                                                            subpop_name,
                                                            subpop_val,
                                                            common_value,
                                                            num_age_groups,
                                                            num_risk_groups)
            params_by_subpop.setdefault(subpop_name, {})[param_name] = value

            if is_age_risk:
                age_risk_param_names.add(param_name)

    if age_risk_param_names and \
            num_age_groups is not None and num_risk_groups is not None:
        _expand_age_risk_fallbacks(params_by_subpop,
                                   age_risk_param_names,
                                   subpop_names,
                                   common_params,
                                   num_age_groups,
                                   num_risk_groups)

    return params_by_subpop


def make_subpop_params_dict(common_params: Union[FilePath, DataClassProtocol],
                            subpop_names: Sequence[str],
                            dataclass_ref: Optional[Type[DataClassProtocol]] = None,
                            subpop_specific_filepath: Optional[FilePath] = None) -> dict:
    """
    Create one parameters instance per subpopulation, starting from
    parameters shared by all subpopulations and applying any
    subpopulation-specific values.

    This is the recommended way to give parameters such as
    `IP_to_ISH_prop` and `ISH_to_HD_prop` -- which are A x R within a
    subpopulation -- values that also vary across the L subpopulations
    of a metapopulation model.

    Args:
        common_params (str | Path | DataClassProtocol):
            either the filepath of the common (shared) parameters
            `JSON` file, or an already-constructed parameters instance.
        subpop_names (Sequence[str]):
            names of the subpopulations to create parameters for -- one
            entry is returned per name, in the given order.
        dataclass_ref (Optional[Type[DataClassProtocol]]):
            parameters dataclass (class, not instance) -- required when
            `common_params` is a filepath. When `common_params` is an
            instance, it defaults to that instance's own class.
        subpop_specific_filepath (Optional[str | Path]):
            filepath of the subpopulation-specific parameters `JSON`
            file -- see `load_subpop_specific_params_json` for its
            format. If `None`, every subpopulation simply gets the
            common parameters. A path that does not exist raises, so
            that a mistyped filename is not silently ignored -- check
            for the file before calling if it is genuinely optional.

    Returns:
        (dict):
            dictionary mapping each name in `subpop_names` to its own
            parameters instance.
    """

    if isinstance(common_params, (str, Path)):
        if dataclass_ref is None:
            raise ValueError(
                "dataclass_ref is required when common_params is a filepath."
            )
        common_params_instance = make_dataclass_from_json(common_params, dataclass_ref)
    else:
        common_params_instance = common_params
        dataclass_ref = dataclass_ref or type(common_params)

    if subpop_specific_filepath is None:
        updates_by_subpop = {}
    else:
        if not Path(subpop_specific_filepath).exists():
            raise FileNotFoundError(
                f"Subpopulation-specific parameters file not found: "
                f"{subpop_specific_filepath}"
            )
        updates_by_subpop = load_subpop_specific_params_json(subpop_specific_filepath,
                                                             subpop_names,
                                                             dataclass_ref,
                                                             common_params_instance)

    return {name: updated_dataclass(common_params_instance,
                                    updates_by_subpop.get(name, {}))
            for name in subpop_names}
