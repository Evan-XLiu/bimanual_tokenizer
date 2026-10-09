import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from vertical_probe import VerticalSource, ACTIVE
from arm_execution import Observation
from quest_bridge import load_quest_profile
from hold_arms import run_hold

class ProbeTests(unittest.TestCase):
    def plan(self):
        return dict(initial_q=[.4]*14,q_path=[[.4+.1*i/20]*14 for i in range(21)])

    def test_changed_start_rejected(self):
        _,config=load_quest_profile()
        obj=SimpleNamespace(observation=Observation((.404,)*14,0,1,5),c=config)
        with self.assertRaisesRegex(ValueError,'Pose changed'):
            VerticalSource(self.plan()).initialize(obj,0)

    def run_probe(self,stop_at=float('inf')):
        profile,config=load_quest_profile();profile['hold_s']=ACTIVE
        class IO:
            t=0.; q=(.4,)*14
            def __init__(self):self.commands=[]
            def check(self):pass
            def observation(self):return Observation(self.q,self.t,round(self.t*10000)+1,5)
            def sleep(self,dt):self.t+=dt
            def write(self,c):self.q=c.q;self.commands.append(c);return True
        io=IO()
        result=run_hold(io,config,profile,lambda:io.t>=stop_at,lambda _:None,
                        lambda:io.t,io.sleep,VerticalSource(self.plan()))
        return io,result

    def test_full_return_and_rates(self):
        io,result=self.run_probe()
        self.assertIsNone(result['error'])
        self.assertGreater(max(c.q[0] for c in io.commands),.4999)
        self.assertAlmostEqual(io.commands[-1].q[0],.4,delta=1e-4)
        self.assertTrue(all(x==0 for x in io.commands[-1].kp+io.commands[-1].kd))
        v=[(b.q[0]-a.q[0])/.01 for a,b in zip(io.commands,io.commands[1:])]
        self.assertLessEqual(max(map(abs,v)),.05+1e-8)
        self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(v,v[1:])),.1+1e-8)

    def test_stop_does_not_continue_or_return(self):
        io,result=self.run_probe(8.)
        self.assertIsNone(result['error'])
        self.assertLess(io.t,11.)
        self.assertGreater(io.commands[-1].q[0],.4)
        self.assertTrue(all(x==0 for x in io.commands[-1].kp))

if __name__=='__main__':unittest.main()
