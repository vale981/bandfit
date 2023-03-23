# Bandfit

This package provides the functionality to fit the band structure of
the long range SSH model to an experimentally measured two dimensional
image.

It is taylored to the specific case where the parameters ``a`` and
``b`` are close to each other.

For documentation, please refer to the docstrings (or the generated
sphinx docs) and check out ``example/example.ipynb``.

Truth be told: this isn't my finest work. The functions are way to
complex for what they do and should be split into smaller pieces. But
it works ;).


## Usage
To develop this package, clone it, install ``poetry`` and run

```shell
$ poetry install --with dev
$ poetry shell
```

This drops you into a virtual enviroment with all the dependencies
installed. The dependencies itself are pretty basic ``(numpy, scipy)``
so just importing the module in you project should also work :).

To use this in production, you can either clone it to somewhere in
your ``PYTHONPATH`` or add it as a git submodule to your source tree,
or just copy the contents of the ``bandfit/bandfit.py`` file.

## Possible Improvements
 - using ``lmfit`` for more accurate parameter error esitmation
 - make the functions less humogous
