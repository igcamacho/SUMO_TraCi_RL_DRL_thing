#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from collections import deque
import time


# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 72000  # number of time steps 7200 orig
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("train_run/data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>


#state_shape=(np.zeros(1), 1)
#state_shape=Box([-20], [20], (2,), int)
state_shape=(None,2)
#action_shape=(np.zeros(30), 1)
action_shape=30

def write_csv(matin,name):
    with open(name, "w") as wtfl:
        df = pd.DataFrame(matin)
        df.to_csv(wtfl, float_format=str, index=False, header=False)

def read_csv(name):
    df = pd.read_csv(name, header=None, index_col=False)
    out=df.to_numpy()
    return out


def agent():
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(48, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(36, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def run_train():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    learning_rate = 0.85 # Learning rate
    discount_factor = 0.618
    decay = 0.01
    Lsup=3
    Linf=2
    cycles=0
    """execute the TraCI control loop"""
    step = 0
    ct=0
    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("0", 2)
    tc1=30
    acl=0
    pre_acl=0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t=traci.simulation.getTime()
        if t==(ct+tc1+6):
            nsc2=traci.lanearea.getLastStepVehicleNumber("e2_22")
            #print("Vehicles waiting at C2="+str(nsc2))
            traci.trafficlight.setPhase("0", 3)
        elif t==(ct+60):
            nsc1=traci.lanearea.getLastStepVehicleNumber("e2_10")+traci.lanearea.getLastStepVehicleNumber("e2_11")
            #print("Vehicles waiting at C1="+str(nsc1))
            traci.trafficlight.setPhase("0", 1)
            ct=t
            dif=nsc1-nsc2
            if dif<-20:
                dif=-20
            elif dif>20:
                dif=20
            if cycles>0:
                if ((dif-Linf)*(dif-Lsup))==0:
                    reward=1
                else:
                    reward=np.power((abs(1/((dif-Linf)*(dif-Lsup)))),1/3)
                if acl>pre_acl:
                    replay_memory.append([observation, action, reward, np.array([nsc1, nsc2]), True])
            random_number=np.random.rand()
            if random_number <= epsilon:
                pass
            else:
                #At this point we have the environment reset, it's time to decide an action
                total_training_rewards = 0
                observation=np.array([nsc1, nsc2])
                encoded=observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                #print(encoded_reshaped)
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
                tc1=action+10
                pre_acl=acl                
                acl=cycles
            cycles+=1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * cycles)
    traci.close()
    sys.stdout.flush()

def run_test():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    learning_rate = 0.001 # Learning rate
    discount_factor = 0.618
    decay = 0.01
    Lsup=3
    Linf=2
    cycles=0
    """execute the TraCI control loop"""
    step = 0
    ct=0
    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("0", 2)
    tc1=30
    acl=0
    pre_acl=0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t=traci.simulation.getTime()
        if t==(ct+tc1+6):
            nsc2=traci.lanearea.getLastStepVehicleNumber("e2_22")
            #print("Vehicles waiting at C2="+str(nsc2))
            traci.trafficlight.setPhase("0", 3)
        elif t==(ct+60):
            nsc1=traci.lanearea.getLastStepVehicleNumber("e2_10")+traci.lanearea.getLastStepVehicleNumber("e2_11")
            #print("Vehicles waiting at C1="+str(nsc1))
            traci.trafficlight.setPhase("0", 1)
            ct=t
            dif=nsc1-nsc2
            if dif<-20:
                dif=-20
            elif dif>20:
                dif=20
            if cycles>0:
                if ((dif-Linf)*(dif-Lsup))==0:
                    reward=1
                else:
                    reward=np.power((abs(1/((dif-Linf)*(dif-Lsup)))),1/3)
                if acl>pre_acl:
                    replay_memory.append([observation, action, reward, np.array([nsc1, nsc2]), True])
            random_number=np.random.rand()
            if random_number <= epsilon:
                pass
            else:
                #At this point we have the environment reset, it's time to decide an action
                total_training_rewards = 0
                observation=np.array([nsc1, nsc2])
                encoded=observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                #print(encoded_reshaped)
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
                tc1=action+10
                pre_acl=acl                
                acl=cycles
            cycles+=1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * cycles)
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# 1. Initialize the Target and Main models
# Main Model (updated every 4 steps)
#model = agent(env.observation_space.shape, env.action_space.n)
model=agent()
# Target Model (updated every 100 steps)
#target_model = agent(env.observation_space.shape, env.action_space.n)
target_model=agent()
target_model.set_weights(model.get_weights())

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    
    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    steps_to_update_target_model = 0

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "train_run/data/cross.sumocfg",
                             "--tripinfo-output", "train_run/tripinfo.xml", "--statistic-output", "train_run/statdrl_train.xml"])
    print("Beginning training")
    run_train()
    generate_routefile()
    traci.start([sumoBinary, "-c", "train_run/data/cross.sumocfg",
                             "--tripinfo-output", "train_run/tripinfo.xml", "--statistic-output", "train_run/statdrl_train.xml"])
    run_train()
    traci.start([sumoBinary, "-c", "test_run/data/cross.sumocfg",
                             "--tripinfo-output", "test_run/tripinfo.xml", "--statistic-output", "test_run/statdrl_test.xml"])
    print("Beginning testing")
    run_test()
    #run()
