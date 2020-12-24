def validate_version_or_image_args(framework_version, py_version):
    """Checks if libary or python version arguments are specified correct.
    Args:
        framework_version (dict): Dictonary with 'transformers' and 'datasets' as key and their version of the library.
        py_version (str): The version of Python.
    Raises:
        ValueError: if `framework_version['transfomers']` is None and `framework_version['datasets']` is None and `py_version` is
            not 3.
    """
    if (
        framework_version is None
        or py_version is not "3"
        or "transformers" not in framework_version
        or "datasets" not in framework_version
    ) and image_uri is None:
        raise ValueError(
            "framework_version or py_version was None, "
            "Either specify both framework_version and py_version, or specify image_uri."
        )
