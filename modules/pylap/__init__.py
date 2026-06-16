# Soft imports — only expose modules that were actually compiled.
# Modules requiring legacy IRI libs (iri2007, iri2012) are absent on PHaRLAP 4.7.4.
_optional = [
    'abso_bg', 'dop_spread_eq', 'ground_bs_loss', 'ground_fs_loss',
    'igrf2007', 'igrf2011', 'igrf2016',
    'iri2007', 'iri2012', 'iri2016',
    'irreg_strength', 'nrlmsise00',
    'raytrace_2d', 'raytrace_2d_sp',
    'raytrace_3d', 'raytrace_3d_sp',
]
for _m in _optional:
    try:
        from importlib import import_module as _imp
        _mod = _imp('pylap.' + _m)
        globals()[_m] = getattr(_mod, _m)
    except (ImportError, AttributeError):
        pass
del _optional, _m, _imp, _mod
