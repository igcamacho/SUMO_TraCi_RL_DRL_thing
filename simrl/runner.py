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

def write_csv(matin,name):
    with open(name, "w") as wtfl:
        df = pd.DataFrame(matin)
        df.to_csv(wtfl, float_format=str, index=False, header=False)

def read_csv(name):
    df = pd.read_csv(name, header=None, index_col=False)
    out=df.to_numpy()
    return out


def run(lr):
    cmat=[0]
    dmat=[]
    Lsup=3
    Linf=2
    cycles=0
    """execute the TraCI control loop"""
    step = 0
    ct=0
    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("0", 2)
    S=-1
    A=-1
    tc1=30
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
            dmat.append(dif)
            #print("Difference="+str(dif))
            r=read_csv("r.csv")
            q=read_csv("q.csv")
            #G_d=np.power((abs(1/((dif-4)*(dif-5)))),1/3)
            S1=dif+20
            if S1>39:
                S1=39
            elif S1<0:
                S1=0
            if ((S==-1) and (A==-1)):
                A1=np.where(np.transpose(q[S1,:])==max(np.transpose(q[S1,:])))
                A1=np.array(A1)[0,0]
                S=S1
                A=A1
            else:
                if ((dif-Linf)*(dif-Lsup))==0:
                    r[S,A]=1
                else:
                    r[S,A]=np.power((abs(1/((dif-Linf)*(dif-Lsup)))),1/3)
                #print(np.transpose(q[S1,:]))
                A1=np.where(np.transpose(q[S1,:])==max(np.transpose(q[S1,:])))
                A1=np.array(A1)[0,0]
                #A1=np.transpose(q[S1,:]).index(max(np.transpose(q[S1,:])))
                q[S,A] = q[S,A]+lr*(r[S,A]+0.55*q[S1,A1]-q[S,A])
                S=S1
                A=A1
                #print(np.array(A)[0,0])
                cycles+=1
                cmat.append(cycles)
            tc1=A+11
            step += 1
            write_csv(q,"q.csv")
            write_csv(r,"r.csv")
    print("There were "+str(cycles)+" traffic light cycles")
    print("Run time was "+str(t)+" seconds")
    plt.plot(cmat, dmat)
    plt.xlabel("Traffic light cycles")
    plt.ylabel("Street occupation difference")
    plt.title("Learning process")
    plt.show()
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    qcsv_exists=os.path.exists("q.csv")
    rcsv_exists=os.path.exists("r.csv")
    if ((not qcsv_exists) or (not rcsv_exists)):
        q_m=np.random.rand(40,30)
        write_csv(q_m,"q.csv")
        q_mr=read_csv("q.csv")
        if np.allclose(q_m,q_mr):
            print("q matrix written OK")
        r_m=np.random.rand(40,30)
        write_csv(r_m,"r.csv")
        r_mr=read_csv("r.csv")
        if np.allclose(r_m,r_mr):
            print("r matrix written OK")


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
                             "--tripinfo-output", "train_run/tripinfo.xml", "--statistic-output", "train_run/train_statrl.xml"])
    run(0.85)
    generate_routefile()
    traci.start([sumoBinary, "-c", "train_run/data/cross.sumocfg",
                             "--tripinfo-output", "train_run/tripinfo.xml", "--statistic-output", "train_run/train_statrl.xml"])
    run(0.85)
    traci.start([sumoBinary, "-c", "test_run/data/cross.sumocfg",
                             "--tripinfo-output", "test_run/tripinfo.xml", "--statistic-output", "test_run/test_statrl.xml"])
    run(0.001)
