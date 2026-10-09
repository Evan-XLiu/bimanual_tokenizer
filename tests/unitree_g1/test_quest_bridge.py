import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import ArmExecution, Observation
from quest_bridge import LIMITS, QuestSource, read_packet, write_json
from test_arm_execution import config
from test_hold_arms import FakeIO
from hold_arms import run_hold
from joint_probe import ElbowProbe


class QuestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.t = 0.
        self.source = QuestSource(self.output, 'new', lambda: self.t)
        self.sent = []
        self.executor = ArmExecution(config(), lambda cmd: self.sent.append(cmd) is None)
        self.executor.observe(Observation((.4,)*14, 0., 1, 5))
        self.packet(phase='WAITING')
        self.source.initialize(self.executor, self.t)
        self.executor.request_control(self.t)

    def packet(self, **overrides):
        value = dict(nonce='new', session='headset', input_time=self.t,
                     computed_time=self.t, state='following', phase='ACTIVE', q=[.42]*14)
        value.update(overrides)
        write_json(self.output/'target.json', value)

    def step(self, packet=True, **overrides):
        self.t += .01
        self.executor.observe(Observation(self.executor.q, self.t, round(self.t*1000)+1, 5))
        if packet:
            self.packet(**overrides)
        self.source.update(self.executor, self.t)
        return self.executor.step(self.t)

    def engage(self):
        for _ in range(10):
            self.step(phase='ENGAGING')
        self.assertEqual(self.executor.phase, 'ACTIVE')

    def test_quest_tracking_threshold_accepts_old_trip_and_still_unloads(self):
        from quest_bridge import load_quest_profile
        profile, cfg = load_quest_profile()
        self.assertEqual(cfg.tracking_error, (.10,)*14)
        self.assertEqual(profile['controller']['tracking_error'], [.10]*14)
        self.assertFalse(profile['hardware_approved'])
        for phase in ('ACTIVE', 'RELEASING'):
            for error, trips in ((.0800014209472461, False), (.099, False), (.101, True)):
                with self.subTest(phase=phase, error=error):
                    sent = []
                    ex = ArmExecution(cfg, lambda cmd: sent.append(cmd) is None)
                    ex.observe(Observation((0.,)*14, 0., 1, 5))
                    ex.request_control(0.)
                    ex.phase, ex.scale = 'ACTIVE', 1.
                    if phase == 'RELEASING':
                        ex.release_control()
                    ex.observe(Observation((error,)+(0.,)*13, .01, 2, 5))
                    ex.step(.01)
                    self.assertEqual(ex.fault, trips)
                    if trips:
                        self.assertEqual(ex.first_fault.reason, 'tracking_error')
                        for step in range(2, 205):
                            now = step*.01
                            ex.observe(Observation((error,)+(0.,)*13, now, step+1, 5))
                            ex.step(now)
                        self.assertEqual(ex.phase, 'FAULT')
                        self.assertEqual(sent[-1].kp, (0.,)*14)

    def test_measured_start_dynamic_bounds_and_rate_limits(self):
        self.assertEqual(self.sent[0].q, (.4,)*14)
        self.assertAlmostEqual(self.executor.c.lower[0], .4-LIMITS['joint_excursion_rad'])
        self.assertAlmostEqual(self.executor.c.upper[0], .4+LIMITS['joint_excursion_rad'])
        self.engage()
        q, v = self.executor.q, (0.,)*14
        for _ in range(100):
            cmd = self.step()
            velocity = tuple((b-a)/.01 for a,b in zip(q,cmd.q))
            self.assertLessEqual(max(abs(x) for x in velocity), LIMITS['velocity_rad_s']+1e-9)
            self.assertLessEqual(max(abs(a-b)/.01 for a,b in zip(velocity,v)), LIMITS['acceleration_rad_s2']+1e-8)
            q, v = cmd.q, velocity
        self.assertGreater(q[0], .41)

    def test_release_during_ramp_or_active_unloads(self):
        for active in (False, True):
            with self.subTest(active=active):
                if active:
                    self.setUp()
                    self.engage()
                self.step(state='ready')
                for _ in range(12):
                    self.t += .01
                    self.executor.observe(Observation(self.executor.q, self.t, round(self.t*1000)+1, 5))
                    self.executor.step(self.t)
                self.assertEqual(self.executor.phase, 'DISARMED')
                self.assertEqual(self.sent[-1].kp, (0.,)*14)

    def test_replayed_or_reconnected_input_fails_closed(self):
        self.engage()
        self.step(session='another')
        self.assertTrue(self.executor.fault)
        self.assertIn('connection changed', self.executor.reason)

    def test_cached_packet_does_not_renew_target(self):
        self.engage()
        self.step()
        accepted_time = self.executor.target_time
        for _ in range(7):
            self.step(packet=False)
        self.assertEqual(self.executor.target_time, accepted_time)
        self.assertTrue(self.executor.fault)
        self.assertEqual(self.executor.first_fault.reason, 'stale_target')

    def test_gripper_targets_require_fresh_active_packet_and_invalid_values_stop_arms(self):
        from quest_grippers import trigger_pair
        from types import SimpleNamespace
        calls=[]
        def update(values,stamp,now):calls.append((trigger_pair(values),stamp))
        self.source.gripper_io=SimpleNamespace(set_gripper_targets=update)
        self.engage()
        self.assertEqual(calls,[])
        self.step(triggers=[.2,.8])
        self.assertEqual(calls[-1][0],(.2,.8))
        self.step(packet=False)
        self.assertEqual(len(calls),1)
        self.step(triggers=None)
        self.assertTrue(self.executor.fault)
        self.assertIn('trigger',self.executor.reason)

    def test_joint_envelope_rejection(self):
        self.engage()
        self.step(q=[.4+LIMITS['joint_excursion_rad']+.01]*14)
        self.assertTrue(self.executor.fault)
        self.assertEqual(self.executor.first_fault.reason, 'invalid_target')
        self.assertTrue(all(c.q == (.4,)*14 for c in self.sent))

    def test_ik_rejection_preserves_cause_instead_of_waiting_for_timeout(self):
        self.engage()
        write_json(self.output/'worker_error.json', {'error':'IK exceeds initial-pose joint envelope'})
        self.step()
        self.assertTrue(self.executor.fault)
        self.assertIn('joint envelope', self.executor.first_fault.reason)

    def test_packet_freshness_nonce_and_finite(self):
        for override in ({'nonce': 'old'}, {'input_time': -1.}, {'computed_time': 1.}):
            self.packet(**override)
            with self.assertRaises(ValueError):
                read_packet(self.output/'target.json', 'new', 0.)

    def test_source_failure_run_loop_unloads(self):
        io = FakeIO()
        source = QuestSource(self.output, 'new', lambda: io.t)
        self.packet(phase='WAITING')
        result = run_hold(io, config(), dict(period_s=.01, hold_s=.2, zero_tail_s=.05),
                          lambda: False, lambda _: None, lambda: io.t, io.sleep, source)
        self.assertIsNotNone(result['error'])
        self.assertTrue(result['unload_submitted'])
        self.assertEqual(io.commands[-1][1].kp, (0.,)*14)


class BoundaryTests(unittest.TestCase):
    def test_free_rotation_is_diagnostic_but_invalid_fk_still_faults(self):
        import numpy as np
        from types import SimpleNamespace
        from quest_bridge import bounded_ik
        reference = [np.eye(4), np.eye(4)]
        rotated = np.eye(4)
        rotated[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        robot = SimpleNamespace(model=SimpleNamespace(
            lowerPositionLimit=np.full(14, -1.), upperPositionLimit=np.full(14, 1.)),
            ik=SimpleNamespace(trial_reference=None,
                smooth_filter=SimpleNamespace(_data_queue=[]),
                opti=SimpleNamespace(set_value=lambda *args: None, stats=lambda: {'success': True}),
                solve_ik=lambda *args: (np.zeros(14), None)),
            forward=lambda: [rotated.copy(), rotated.copy()])
        evidence = {}
        bounded_ik(robot, reference, reference, reference,
                   dict(q=[0.]*14, initial_q=[0.]*14), evidence)
        self.assertEqual(evidence['accepted_fraction'], 1.)
        self.assertEqual(evidence['orientation_mode'], 'free')
        self.assertGreater(min(evidence['ik_rotation_change_rad']), 1.)
        rotated[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite IK'):
            bounded_ik(robot, reference, reference, reference,
                       dict(q=[0.]*14, initial_q=[0.]*14), {})

    def test_workspace_projection_backtracking_return_and_faults(self):
        import numpy as np
        from types import SimpleNamespace
        from quest_bridge import bounded_ik
        references = [np.eye(4), np.eye(4)]
        feedback = dict(q=[0.]*14, initial_q=[0.]*14)
        robot = SimpleNamespace(q=np.zeros(14))
        robot.model = SimpleNamespace(lowerPositionLimit=np.full(14, -1.),
                                      upperPositionLimit=np.full(14, 1.))
        robot.ik = SimpleNamespace(trial_reference=None,
            smooth_filter=SimpleNamespace(_data_queue=[]),
            opti=SimpleNamespace(set_value=lambda *args: None, stats=lambda: {'success': True}))
        def solve(left, right, *args):
            robot.achieved = [left.copy(), right.copy()]
            # Toy arm is reachable only up to 4 cm; solver obeys joint bounds.
            for t in robot.achieved:
                t[0, 3] = min(t[0, 3], .04)
            return np.zeros(14), None
        robot.ik.solve_ik = solve
        robot.forward = lambda: robot.achieved
        requested = [t.copy() for t in references]
        requested[0][:3, 3] = [.6, .1, .1]
        evidence = {}
        q, errors, accepted = bounded_ik(robot, requested, references, references, feedback, evidence)
        self.assertEqual(evidence['workspace_limited'], [True, False])
        self.assertAlmostEqual(evidence['workspace_target'][0][0][3], .1)
        self.assertAlmostEqual(evidence['workspace_target'][0][1][3], .1)
        self.assertAlmostEqual(evidence['workspace_target'][0][2][3], .1)
        self.assertTrue(evidence['ik_limited'])
        self.assertLessEqual(max(errors), .015)
        self.assertGreater(accepted[0][0, 3], 0.)
        self.assertAlmostEqual(requested[0][0, 3], .6)  # raw input preserved
        returned = {}
        bounded_ik(robot, references, references, accepted, feedback, returned)
        self.assertFalse(returned['ik_limited'])
        self.assertEqual(returned['accepted_fraction'], 1.)
        below = [t.copy() for t in references]
        below[0][2, 3] = -.1
        floor_evidence = {}
        bounded_ik(robot, below, references, references, feedback, floor_evidence)
        self.assertTrue(floor_evidence['workspace_limited'][0])
        self.assertEqual(floor_evidence['workspace_target'][0][2][3], 0.)
        robot.ik.opti.stats = lambda: {'success': False}
        with self.assertRaisesRegex(ValueError, 'IK failed'):
            bounded_ik(robot, requested, references, accepted, feedback, {})
        robot.ik.opti.stats = lambda: {'success': True}
        robot.ik.solve_ik = lambda *args: (np.ones(14), None)
        with self.assertRaisesRegex(ValueError, 'joint constraints'):
            bounded_ik(robot, requested, references, accepted, feedback, {})

    def test_no_valid_fallback_is_fault(self):
        import numpy as np
        from types import SimpleNamespace
        from quest_bridge import bounded_ik
        robot = SimpleNamespace(q=np.zeros(14))
        robot.model = SimpleNamespace(lowerPositionLimit=np.full(14, -1.),
                                      upperPositionLimit=np.full(14, 1.))
        robot.ik = SimpleNamespace(trial_reference=None,
            smooth_filter=SimpleNamespace(_data_queue=[]),
            opti=SimpleNamespace(set_value=lambda *args: None, stats=lambda: {'success': True}),
            solve_ik=lambda *args: (np.zeros(14), None))
        bad = np.eye(4)
        bad[0, 3] = 1.
        robot.forward = lambda: [bad, bad]
        evidence = {}
        with self.assertRaisesRegex(ValueError, 'cannot revalidate'):
            bounded_ik(robot, [np.eye(4)]*2, [np.eye(4)]*2, [np.eye(4)]*2,
                       dict(q=[0.]*14, initial_q=[0.]*14), evidence)
        self.assertEqual(len(evidence['attempts']), 5)


class IKWorkerTests(unittest.TestCase):
    def test_position_only_five_cm_lift_and_controller_rotation_independence(self):
        code = r'''
import tempfile
from pathlib import Path
import numpy as np
from virtual_arms import VirtualRobot
from quest_bridge import configure_measured_pose_ik, bounded_ik
# Recorded preflight posture that could not satisfy the fixed-direction task.
initial=np.array([-.0695804134,.1462914348,-.0770226121,.7641501427,-.231343478,
                  -.68005687,-.0306436643,-.0190549176,-.086837694,.0838536248,
                  .6517500877,.2000047415,-.60613817,-.0661529228])
with tempfile.TemporaryDirectory() as tmp:
    robot=VirtualRobot(Path(tmp),False)
    configure_measured_pose_ik(robot)
    robot.q=initial.copy()
    references=robot.forward()
    previous=references
    seed=initial.copy()
    for height in np.linspace(.0025,.05,20):
        targets=[x.copy() for x in references]
        for t in targets: t[2,3]+=height
        feedback=dict(q=seed.tolist(),initial_q=initial.tolist())
        evidence={}
        q,errors,accepted=bounded_ik(robot,targets,references,previous,feedback,evidence)
        assert evidence['accepted_fraction']==1., evidence
        assert max(errors)<.003, errors
        assert np.max(np.abs(np.asarray(q)-initial))<=.2
        rotated=[x.copy() for x in targets]
        for t in rotated:
            t[:3,:3]=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]) @ t[:3,:3]
        q_rot,_,_=bounded_ik(robot,rotated,references,previous,feedback,{})
        assert np.max(np.abs(np.asarray(q)-q_rot))<1e-5
        seed=np.asarray(q); previous=accepted
    assert min(evidence['ik_rotation_change_rad'])>.1
'''
        subprocess.run([sys.executable, '-c', code],
                       cwd=Path(__file__).resolve().parents[2]/'scripts/unitree_g1', check=True, timeout=30)

    def test_ik_solver_respects_initial_pose_envelope(self):
        code = r'''
import tempfile
from pathlib import Path
import numpy as np
from virtual_arms import VirtualRobot
from quest_bridge import configure_measured_pose_ik, LIMITS
with tempfile.TemporaryDirectory() as tmp:
    robot=VirtualRobot(Path(tmp),False)
    configure_measured_pose_ik(robot)
    initial=np.array([.05,.04,.2,1.25,.1,.04,0.,-.05,.04,.1,1.25,.1,.04,-.1])
    robot.q=initial.copy()
    reference=robot.forward()
    robot.ik.opti.set_value(robot.ik.trial_reference,initial)
    for dz in (0.,.002,.025):
        target=[m.copy() for m in reference]
        target[1][2,3]+=dz
        robot.ik.smooth_filter._data_queue=[]
        candidate,_=robot.ik.solve_ik(*target,initial.copy(),np.zeros(14))
        assert robot.ik.opti.stats()['success']
        assert np.max(np.abs(candidate-initial)) <= LIMITS['joint_excursion_rad']
        if dz == 0:
            assert np.max(np.abs(candidate-initial)) < 1e-4
'''
        subprocess.run([sys.executable, '-c', code],
                       cwd=Path(__file__).resolve().parents[2]/'scripts/unitree_g1', check=True, timeout=30)

    def test_measured_pose_and_worker_release_latch(self):
        # Separate process preserves the existing SDK import-isolation tests.
        code = r'''
import json, multiprocessing, tempfile, time
from pathlib import Path
from quest_bridge import worker, write_json

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp)
    (output/'input').mkdir()
    ctx = multiprocessing.get_context('spawn')
    done = ctx.Event()
    proc = ctx.Process(target=worker, args=(output, 'test', done, True))
    proc.start()
    try:
        deadline = time.monotonic()+20
        while not (output/'ik_ready.json').exists():
            assert proc.is_alive(), (output/'worker_error.json').read_text()
            assert time.monotonic() < deadline
            time.sleep(.01)
        q = [.05,.04,.2,1.25,.1,.04,0.,-.05,.04,.1,1.25,.1,.04,-.1]
        phase = 'WAITING'
        def feedback():
            write_json(output/'feedback.json', dict(nonce='test', time=time.monotonic(),
                q=q, command_q=q, initial_q=q, phase=phase))
        def event(grip, x=0.):
            now = time.monotonic()
            pose = [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,x,1.]
            data = dict(kind='CONTROLLER_MOVE',session='head',valid=True,host_receive_monotonic_s=now,
                raw={'left':pose,'right':pose,'leftState':{'squeezeValue':grip,'triggerValue':.2},'rightState':{'squeezeValue':grip,'triggerValue':.8}})
            feedback()
            with (output/'input/events.jsonl').open('a') as f:
                f.write(json.dumps(data)+'\n')
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                feedback()
                assert not (output/'worker_error.json').exists(), (output/'worker_error.json').read_text() if (output/'worker_error.json').exists() else ''
                if (output/'target.json').exists():
                    packet = json.loads((output/'target.json').read_text())
                    if packet['input_time'] == now:
                        return packet
                time.sleep(.002)
            raise AssertionError('worker did not respond')
        assert event(0.)['state'] == 'ready'
        assert event(1.)['state'] == 'following'
        phase = 'ENGAGING'
        event(1.)
        phase = 'ACTIVE'
        event(1.)
        neutral = event(1.)
        assert neutral['triggers']==[.2,.8], neutral
        assert max(abs(a-b) for a,b in zip(neutral['q'],q)) < 1e-4, neutral
        moved = event(1., -.01)
        assert max(abs(a-b) for a,b in zip(moved['q'],q)) > .001, moved
        assert max(moved['ik_position_error_m']) < .015
        limited = False
        for x in [-.05*i for i in range(1, 13)]:
            packet = event(1., x)
            assert packet['state'] == 'following', packet
            assert max(packet['ik_position_error_m']) <= .015, packet
            limited |= any(packet['ik_evidence']['workspace_limited'])
        assert limited
        for x in [-.05*i for i in range(11, -1, -1)]:
            packet = event(1., x)
            assert packet['state'] == 'following'
        assert not any(packet['ik_evidence']['workspace_limited'])
        stopped = event(0.)
        assert stopped['state'] == 'ready'
        before = (output/'target.json').read_bytes()
        time.sleep(.05)
        assert (output/'target.json').read_bytes() == before
    finally:
        done.set()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
            proc.join()
'''
        subprocess.run([sys.executable, '-c', code],
                       cwd=Path(__file__).resolve().parents[2]/'scripts/unitree_g1', check=True, timeout=30)


class ProbeTests(unittest.TestCase):
    def test_only_elbow_moves_and_returns_with_bounded_rates(self):
        io = FakeIO()
        result = run_hold(io, config(), dict(period_s=.01,hold_s=6.,zero_tail_s=.05),
            lambda:False, lambda _:None, lambda:io.t, io.sleep, ElbowProbe())
        self.assertIsNone(result['error'])
        positions = [c.q for _,c in io.commands]
        self.assertGreater(max(q[3] for q in positions), .419)
        self.assertLessEqual(max(q[3] for q in positions), .425)
        self.assertAlmostEqual(positions[-1][3], .4, delta=.0001)
        for q in positions:
            self.assertTrue(all(x == .4 for i,x in enumerate(q) if i != 3))
        velocities = [(b[3]-a[3])/.01 for a,b in zip(positions,positions[1:])]
        self.assertLessEqual(max(abs(v) for v in velocities), .01+1e-9)
        # Stop freezes the final target; smoothness is required during motion.
        self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(velocities,velocities[1:])), .02+1e-8)


if __name__ == '__main__':
    unittest.main()
