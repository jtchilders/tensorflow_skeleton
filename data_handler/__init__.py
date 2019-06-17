import logging
logger = logging.getLogger('data_handler')

def get_datasets(config):

   if 'random_file_generator' in config['data']['handler']:
      logger.info('using data handler %s',config['data']['handler'])
      import data_handler.random_file_generator as handler
   elif 'h5_file_generator' in config['data']['handler']:
      logger.info('using data handler %s',config['data']['handler'])
      import data_handler.h5_file_generator as handler

   return handler.get_datasets(config)

