mkdir data && cd data

curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
uvx hf auth login
uv init --python 3.13

uv add boto3 botocore[crt] matplotlib numpy requests tiktoken tqdm pandas pyarrow huggingface_hub omegaconf loguru
sudo yum install htop -y

aws login

aws configure set region us-east-1
aws configure
aws configure list
