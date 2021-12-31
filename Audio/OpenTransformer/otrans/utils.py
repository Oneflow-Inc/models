import logging


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


def count_parameters(named_parameters):
    # Count total parameters
    total_params = 0
    part_params = {}
    for name, p in sorted(list(named_parameters)):
        n_params = p.numel()
        total_params += n_params
        logging.debug("%s %d" % (name, n_params))
        part_name = name.split('.')[0]
        if part_name in part_params:
            part_params[part_name] += n_params
        else:
            part_params[part_name] = n_params
    
    for name, n_params in part_params.items():
        logging.info('%s #params: %d' % (name, n_params))
    logging.info("Total %.2f M parameters" % (total_params / 1000000))
    logging.info('Estimated Total Size (MB): %0.2f' % (total_params * 4. /(1024 ** 2)))