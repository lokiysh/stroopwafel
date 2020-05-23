import os
import glob
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "modules"))
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [ os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not (f.endswith('__init__.py') or f.endswith('find_rejection_rate.py') \
    or f.endswith('find_probability_of_locations_from_distribution.py'))]