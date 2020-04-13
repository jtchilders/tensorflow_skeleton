import logging
logger = logging.getLogger('model')

__all__ = ['mnist','resnet']
from . import mnist,resnet


def get_model(config):
   if config['model']['name'] in globals():
      logger.info('using model name %s',config['model']['name'])
      model = globals()[config['model']['name']]
   else:
      raise Exception('failed to find model handler %s ' % config['model']['name'])

   return model.get_model(config)
