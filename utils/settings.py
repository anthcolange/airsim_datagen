"""
The implementation of settings class
Support:
	- load default settings from config/xxx.py file
	- override settings by command line
	- print out configuration

Author: Jianxiong Cai
"""

import os.path
import importlib
import time

class Settings:
	def __init__(self, cfg_filename):
		"""

		:param cfg_filename: The cfg filename (without extension)
		"""
		# Make sure cfg filename is valid
		if len(cfg_filename.split('.')) != 1:
			raise RuntimeError("[ERROR] Invalid config filename. (config filename does not contain '.')")

		# load the config
		m = importlib.import_module("config."+cfg_filename)
		for k, v in m.cfg.items():
			setattr(self, k, v)


	def parse_cmd_args(self, args):
		"""
		Override settings by command line arguments
		:param args: from ArgParser.parse()
		:return: None
		"""
		for k, v in vars(args).items():
			if k == "cfg":
				continue				# reserved keyword (for loading settings)

			if v is None:
				continue				# None stands for not set

			if not hasattr(self, k):
				raise RuntimeError("[ERROR] Not recognized argument from comamnd line: {}.".format(k))
			setattr(self, k, v)

	def print_args(self, delay_time=1):
		"""

		:param delay_time: time to wait (e.g. 1). Make sure the parameters are printed before moving on
			To avoid messing up logging.
		:return:
		"""
		print("================ Arguments ==================")
		for k, v in vars(self).items():
			print("{} : {}".format(k, v))
		print("=============================================")

		# wait for 1 second (for printing)
		time.sleep(delay_time)

	def dump_to_log(self, log_dir, log_filename):
		"""
		Save experiment configuration to dump config.txt under the log directory
		Input:
		@param log_dir: the directory for dumping the config
		@param log_filename: the filename to save the setting config

		TODO(jianxiong): add a saved filename so that it can be shared with different utils
		"""
		with open(os.path.join(log_dir, log_filename), "w") as f:
			f.write("================ Arguments ==================\n")
			for k, v in vars(self).items():
				f.write("{} : {}\n".format(k, v))
			f.write("=============================================\n")

