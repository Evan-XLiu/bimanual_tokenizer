"""File-based routing of one persistent VR input service to isolated attempts."""
import json
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n')
    temporary.replace(path)


class RoutedEvents:
    """Only the input service owns event files; idle/rejected input is not retained."""
    def __init__(self, service):
        self.service = Path(service)
        self.route = None
        self.stream = None
        self.output = None

    def sync(self):
        route = json.loads((self.service/'route.json').read_text())
        if route == self.route:
            return
        self.close()
        self.output = None
        if route:
            output = Path(route['output']).resolve()
            pending = (self.service.parent/'pending').resolve()
            if not output.is_relative_to(pending) or output.name != 'input' or not output.parent.is_dir():
                raise ValueError('VR route must point to the current pending attempt/input')
            if not route.get('nonce') or not isinstance(route.get('since'), (int,float)):
                raise ValueError('VR route requires nonce and start time')
            output.mkdir(exist_ok=False)
            (output/'session.json').write_bytes((self.service/'session.json').read_bytes())
            self.stream = (output/'events.jsonl').open('x',buffering=1)
            self.output = output
        self.route = route
        write_json(self.service/'route_status.json',dict(route=route,time=time.monotonic()))

    def write(self, payload):
        self.sync()
        event = json.loads(payload)
        if self.stream is not None and event['host_receive_monotonic_s'] >= self.route['since']:
            self.stream.write(payload+'\n')

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def check_service(service, max_age=2.5):
    status = json.loads((Path(service)/'status.json').read_text())
    if not 0 <= time.monotonic()-status['time'] <= max_age or status.get('queue_dropped',0):
        raise RuntimeError('Persistent VR service stale or dropped input')
    return status


def set_route(service, output=None, nonce=None, timeout=5.):
    service = Path(service)
    route = {} if output is None else dict(output=str(Path(output).resolve()),nonce=nonce,since=time.monotonic())
    write_json(service/'route.json',route)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        check_service(service)
        try:
            status = json.loads((service/'route_status.json').read_text())
            if status['route'] == route:
                return
        except FileNotFoundError:
            pass
        time.sleep(.02)
    raise TimeoutError('Persistent VR service did not acknowledge route change')


def routed_gripper_status(service):
    try:
        route = json.loads((Path(service)/'route.json').read_text())
        return Path(route['output']).parent/'gripper_status.json' if route else None
    except (OSError,ValueError,KeyError):
        return None
