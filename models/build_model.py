from .model_simple import ModelSimple

from trainer.names import PHYSICAL_DIMS_SCALAR, PHYSICAL_DIMS, SOLUTION_DIMS

models = dict(
    simple=ModelSimple
)


def get_target_dim_for(head_type):
    if head_type == 'de':
        return SOLUTION_DIMS
    elif head_type == 'vector':
        return PHYSICAL_DIMS
    elif head_type == 'scalar':
        return PHYSICAL_DIMS_SCALAR
    else:
        raise (Exception("{} is not a headtype model (choose among 'de', 'vector', 'scalar')".format(head_type)))


def get_model(name):
    if name in models.keys():
        return models[name]
    else:
        raise (Exception("{} not a valid model (choose among {})".format(name, models.keys())))


def build_model(name, feature_dim, head_type, filters, kernels):
    target_dim_dict = get_target_dim_for(head_type)
    Model = get_model(name)
    model = Model(feature_dim, target_dim_dict, filters, kernels).build()

    return model