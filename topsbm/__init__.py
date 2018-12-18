
try:
    # This variable is injected in builtins by setup.py
    # It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __IN_SETUP__
except NameError:
    __IN_SETUP__ = False

if not __IN_SETUP__:
    from .transformer import TopSBM
    __all__ = ['TopSBM']


__version__ = '0.1.dev1'
