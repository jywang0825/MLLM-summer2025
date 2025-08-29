"""
Automatically block bitsandbytes imports to prevent PEFT errors
This file is loaded by Python at startup when PYTHONPATH includes this directory
"""

import sys
from unittest.mock import MagicMock

# Block bitsandbytes imports before anything else
class MockBitsAndBytesModule:
    def __init__(self):
        self.nn = MagicMock()
        self.nn.Linear4bit = MagicMock()
        self.nn.Linear8bitLt = MagicMock()
        self.optim = MagicMock()
        self.optim.GlobalOptimManager = MagicMock()
        
    def __getattr__(self, name):
        return MagicMock()

# Create the mock module and insert it into sys.modules
mock_bnb = MockBitsAndBytesModule()
sys.modules['bitsandbytes'] = mock_bnb
sys.modules['bitsandbytes.nn'] = mock_bnb.nn
sys.modules['bitsandbytes.optim'] = mock_bnb.optim

print("bitsandbytes blocked automatically via sitecustomize.py") 