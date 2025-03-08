nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 0 -algo FedMoEKD -lam 10 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedMoEKD.log 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 1 -algo pFedMoE -lam 10 > ../logs/total-Cifar10-dir-CNNs-fd=512-pFedMoE.log 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 0 -algo FedProto -lam 10 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedProto.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 1 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedGen.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 0 -algo FedDistill -lam 1 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedDistill.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 1 -algo FML -al 0.5 -bt 0.5 > ../logs/total-Cifar10-dir-CNNs-fd=512-FML.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 0 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedKD.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 1 -algo LG-FedAvg > ../logs/total-Cifar10-dir-CNNs-fd=512-LG-FedAvg.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 0 -algo FedGH -slr 0.01 -se 1 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedGH.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNNs -fd 512 -did 1 -algo FedTGP -lam 10 -se 100 -mart 100 > ../logs/total-Cifar10-dir-CNNs-fd=512-FedTGP.out 2>&1
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 0 -algo FedMoEKD -lam 10 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedMoEKD.log 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 1 -algo pFedMoE -lam 10 > ../logs/total-Cifar10-dir-CNN_1-fd=512-pFedMoE.log 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 0 -algo FedProto -lam 10 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedProto.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 1 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedGen.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 0 -algo FedDistill -lam 1 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedDistill.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 1 -algo FML -al 0.5 -bt 0.5 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FML.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 0 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedKD.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 1 -algo LG-FedAvg > ../logs/total-Cifar10-dir-CNN_1-fd=512-LG-FedAvg.out 2>&1 &
wait
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 0 -algo FedGH -slr 0.01 -se 1 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedGH.out 2>&1 &
nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 10 -nb 10 -data Cifar10 -m CNN_1 -fd 512 -did 1 -algo FedTGP -lam 10 -se 100 -mart 100 > ../logs/total-Cifar10-dir-CNN_1-fd=512-FedTGP.out 2>&1