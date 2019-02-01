import json
import os


def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """

    def normalize_path(*args, **kwargs):
        return os.path.normpath(path(*args, **kwargs))

    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """

    def assure_exists(*args, **kwargs):
        p = path(*args, **kwargs)
        assert os.path.exists(p), "the following path does not exist: '{}'".format(p)
        return p

    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """

    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args, **kwargs):
        if not os.path.exists(path(*args, **kwargs)):
            "Path does not exist... creating it: {}".format(path(*args, **kwargs))
            os.makedirs(path(*args, **kwargs))
        return path(*args, **kwargs)

    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """

    @_norm_path
    @_assure_path_exists
    def check_existence(*args, **kwargs):
        return path(*args, **kwargs)

    return check_existence


@_is_output_path
def get_data_path():
    path = "./data/"
    return path


@_is_output_path
def get_training_data_path(data_version: str):
    path = os.path.join(get_data_path(), data_version, "train")
    return path


@_is_output_path
def get_augmented_data_path(data_version: str):
    path = os.path.join(get_data_path(), data_version, "augmented")
    return path


@_is_output_path
def get_tensorboard_logs_path():
    path = json.load(open("settings.json"))["paths"]["tensorboard_logs"]
    return path


@_is_output_path
def get_models_path():
    path = "./models/"
    return path


@_is_output_path
def get_model_path(model_alias: str):
    path = os.path.join(get_models_path(), "{0}".format(model_alias))
    return path


@_is_output_path
def get_outputs_path():
    path = "./outputs/"
    return path


@_is_output_path
def get_output_path(model_alias: str):
    path = os.path.join(get_outputs_path(), "{0}".format(model_alias))
    return path


def get_dataset_filepath(data_version: str):
    path = os.path.join(get_training_data_path(data_version), "speech_commands.tar.gz")
    return path
