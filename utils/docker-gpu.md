# Пример установки docker для работы с GPU на Ubuntu 22.04

```bash
add-apt-repository ppa:graphics-drivers/ppa -y
 
wget https://nvidia.github.io/nvidia-docker/gpgkey --no-check-certificate
apt-key add gpgkey
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
 
 
 
for kernel in $(linux-version list); do
  apt install -y "linux-headers-${kernel}"
done
 
apt install alsa-utils -y
apt install ubuntu-drivers-common -y
 
ubuntu-drivers devices
 
apt install -y nvidia-driver-550 nvidia-container-toolkit
 
mkdir -p /etc/docker/
 
cat <<EOF > /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
      "nvidia": {
        "path": "/usr/bin/nvidia-container-runtime",
        "runtimeArgs": []
      }
  },
  "default-address-pools":
  [
    {
      "base":"10.10.0.0/16","size":24
    }
  ]
}
EOF
 
 
# Add Docker's official GPG key:
apt-get update
apt-get install ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
 
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
 
apt-get update
add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu jammy stable"
 
apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin  # apt update
 
# apt install docker-ce=5:24.0.7-1~ubuntu.22.04~jammy -y
 
systemctl enable docker
systemctl start docker
```