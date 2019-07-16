import pytest
from antu.io.configurators import IniConfigurator
import os


class TestIniConfigurator:

    def setup(self):
        with open('tmp_test_ini_configurator.ini', 'w') as f:
            test = [
                "[Test1]\n",
                "A = 123 \n",
                "B = 1.1 \n",
                "[Test2]\n",
                "C = add\n",
                "E = 1+3\n",
                "D = 7*2\n",
                "F = %(E)s*2\n",
                "G = %(C)sdda\n",
            ]
            f.writelines(test)

    def test_ini_configurator(self):
        cfg = IniConfigurator('tmp_test_ini_configurator.ini')
        assert cfg.A == 123
        assert cfg.B == 1.1
        assert cfg.C == 'add'
        assert cfg.E == 4
        assert cfg.D == 14
        assert cfg.F == 7
        assert cfg.G == 'adddda'

    def teardown(self):
        tmp_file = 'tmp_test_ini_configurator.ini'
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
