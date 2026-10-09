#!/usr/bin/env python3
"""Query mode services only; never select/release a mode or publish motor commands."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    ChannelFactoryInitialize(0, args.interface)
    result = {'read_only': True, 'interface': args.interface}
    for name, client_type, method in [('motion_service', MotionSwitcherClient, 'CheckMode'),
                                      ('loco_fsm', LocoClient, 'GetFsmId')]:
        client = client_type()
        client.SetTimeout(3.0)
        client.Init()
        try:
            code, value = getattr(client, method)()
            result[name] = {'code': code, 'value': value}
        except Exception as exc:
            result[name] = {'error': str(exc)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
