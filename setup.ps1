pyenv install 3.11.7
pyenv local 3.11.7
pip install virtualenv
virtualenv INM706_CW_env
INM706_CW_env/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118