#!/bin/bash

echo "--------------------------"
echo "install packages for build"
echo "--------------------------"
apt-get upgrade -y
apt-get update -y
apt-get install git -y
apt-get install curl -y
apt-get install gcc make -y
apt-get install -y net-tools tree vim telnet netcat
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
apt-get install -y mysql-server

echo "-------------------------------"
echo "install pyenv and bashrc update"
echo "-------------------------------"
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
if !(grep -qc "PYENV_ROOT" ~/.bashrc); then
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
fi

sleep 5
. ~/.bashrc

echo "--------------------"
echo "install python3.11.4"
echo "--------------------"
pyenv install 3.11.4
pyenv global 3.11.4

echo "--------------------------------"
echo "install poetry and bashrc update"
echo "--------------------------------"
curl -sSL https://install.python-poetry.org | python3 -
if !(grep -qc "$HOME/.local/bin" ~/.bashrc); then
	echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "Executing ~/.bashrc" >> ~/.bashrc
. ~/.bashrc

poetry config virtualenvs.in-project true