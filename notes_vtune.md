## Intel VTune (GUI)

### vtune-gui with VNC

- Create docker image:
```
cd docker_vnc
docker build -t vtune-gui . -f Dockerfile
cd ..
```

- Run docker container:

```
docker run --rm -itd \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --device=/dev/dri \
    --name=vtune-gui-container \
    -v $PWD:/ws \
    -w /ws \
    -v /home/user/Documents/ml/pytorch/:/pytorch \
    --network=host \
    --security-opt seccomp:unconfined \
    --ipc=host \
    vtune-gui
```

```
docker exec -it vtune-gui-container /bin/bash
```

- Run profiling as "Remote Linux (SSH)"
```
# in vtune container:
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub

# in pth-dev container:
apt-get install openssh-server
nano ~/.ssh/authorized_keys
```

- Run perf snapshot
    - python check_interp.py --use-perf
    - sampling interval 0.01 ms


- Run hotspot analysis




## Run linux perf

- Set on host:
```
cat /proc/sys/kernel/perf_event_paranoid
> 4

sudo su
echo "-1" > /proc/sys/kernel/perf_event_paranoid

cat /proc/sys/kernel/perf_event_paranoid
> -1
```

- Run linux perf:

// (https://www.slideshare.net/emBO_Conference/profiling-your-applications-using-the-linux-perf-tools)
// perf record --call-graph dwarf -F 40 -- ./bench 0 0 0 1
// perf record --call-graph dwarf -F 40 -- ./bench 1 0 0 1
// perf report

```
perf record --call-graph dwarf -F 40 -- python -u check_interp.py --use-perf
perf report
```
