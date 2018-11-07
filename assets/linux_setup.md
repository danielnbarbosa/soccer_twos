## Setup
Training can be run in AWS to take advantage of cloud GPUs.  Launch Templates are used to automate a lot of the instance setup.  The below will create an instance based on Unity's ML Agents AMI, which already has most of what we need for running Unity ML environments in AWS.


#### Step 1: Launch Spot Instance
- Use Launch Template to start instance (see below for setup)


#### Step 2: On Every SSH
```
source activate pytorch_p36
export DISPLAY=:0
```



## Launch Template
- Region: us-east-1
- AMI ID: ami-18642967
- Instance Type: p2.xlarge
- Security Group: ssh-inbound-only  (you'll need to create your own)
- Key Pair: daniel  (you'll need to create your own)
- Purchasing option: spot
- User Data (see below)


#### User Data Script

```
#!/bin/bash

/usr/bin/X :0 &
su - -c "source activate pytorch_p36; pip install torchsummary tensorboardX unityagents" ubuntu
DIR="/home/ubuntu"

# install ml agents toolkit
su -c "cd $DIR/ml-agents; git pull" ubuntu
su - -c "source activate pytorch_p36; cd $DIR/ml-agents/ml-agents; pip install ." ubuntu

# update packages
su - -c "conda update -y -n pytorch_p36 --all" ubuntu
```
