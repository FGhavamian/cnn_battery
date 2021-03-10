from .model_simple import ModelSimple

from trainer.utils import get_target_dim_for

models = dict(
    simple=ModelSimple
)

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