"""Exercise the bounded remote recovery failure path without robot access."""
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from dex1_service import ensure_service


class ServiceTests(unittest.TestCase):
    def test_repair_changes_full_unit_and_preserves_backup_on_repeat(self):
        source=(Path(__file__).resolve().parents[2]/'scripts/unitree_g1/install_dex1_service.sh').read_text()
        # Execute the actual file-editing part against a temporary filesystem.
        edits=source[source.index('unit=/etc/'):source.index('# Also reload')]
        original=('[Unit]\nDescription=Dex1\nAfter=multi-user.target\n'
                  '[Service]\nUser=unitree\nExecStart=/robot/gripper\nRestart=always\n'
                  '[Install]\nWantedBy=multi-user.target\n')
        with tempfile.TemporaryDirectory() as tmp:
            unit=Path(tmp)/'dex1_gripper.service'
            directory=Path(str(unit)+'.d')
            unit.write_text(original)
            directory.mkdir()
            (directory/'90-g1-startup.conf').write_text('[Unit]\nAfter=\nAfter=basic.target\n')
            edits=edits.replace('unit=/etc/systemd/system/dex1_gripper.service',
                                'unit='+shlex.quote(str(unit)))
            edits=edits.replace('dir=/etc/systemd/system/dex1_gripper.service.d',
                                'dir='+shlex.quote(str(directory)))
            edits=edits.replace('-o root -g root ','')
            for _ in range(2):
                subprocess.run(['sh','-eu','-c',edits],check=True,capture_output=True,text=True)
                self.assertEqual(unit.read_text(),original.replace(
                    'After=multi-user.target','After=basic.target systemd-udev-trigger.service'))
                self.assertEqual((directory/'original-unit.backup').read_text(),original)
                self.assertNotIn('After=\n',(directory/'90-g1-startup.conf').read_text())

    def test_missing_privilege_stops_with_install_instruction(self):
        result=subprocess.CompletedProcess([],1,'','sudo: a password is required')
        with patch('dex1_service.subprocess.run',return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError,'--install'):
                ensure_service('unitree@robot',Path('/key'))
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.args[0][-1],'sudo -n /usr/local/sbin/g1-ensure-dex1')

    def test_success_retains_service_evidence(self):
        result=subprocess.CompletedProcess([],0,'ActiveState=active\n','')
        with patch('dex1_service.subprocess.run',return_value=result):
            self.assertEqual(ensure_service('unitree@robot',Path('/key'))['details'],'ActiveState=active')
