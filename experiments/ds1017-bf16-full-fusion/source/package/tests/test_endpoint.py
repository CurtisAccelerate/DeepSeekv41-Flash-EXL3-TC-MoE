import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from control_client import endpoint

class EndpointTests(unittest.TestCase):
    def test_explicit_launch_port_drives_client(self):
        self.assertEqual(endpoint(dict(port=8097,cmd=['--port','8097'])),'http://127.0.0.1:8097')
    def test_ambiguous_or_invalid_port_rejected(self):
        for port,args in [(True,['--port','1']),(80,['--port','80']),(8097,['--port','8096']),(8097,[])]:
            with self.assertRaises(ValueError):endpoint(dict(port=port,cmd=args))
