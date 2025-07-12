#!/usr/bin/env python3

from jtop import jtop
import pprint

'''Debugging script to check jtop utility on Jetson Nano.'''

# Only run one sample for simplicity
with jtop() as jetson:
    if not jetson.ok():
        print("Could not connect to jtop.")
        exit(1)

    stats = jetson.stats

    print("========== Jetson Stats ==========")
    pprint.pprint(stats)
    print("==================================")
