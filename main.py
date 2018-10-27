#!/usr/bin/env python

"""
Main program.
"""

import argparse
import environment
import agent
import training

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="path to saved model", type=str, default=None)
parser.add_argument("--eval", help="run in evaluation mode.  no training.  no noise.", action="store_true")
args = parser.parse_args()

# create environment and agent, then run training
environment = environment.UnityMLVectorMultiAgent(evaluation_only=args.eval)
agent = agent.MADDPG(load_file=args.load, evaluation_only=args.eval)
training.train(environment, agent)
