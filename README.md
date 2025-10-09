```
git clone https://github.com/mminggoo/test.git
cd test
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install -r mmada/requirements.txt
```

```
python mmada/mmada_m3cot5.py --pos_penalty_gamma 0.5 --vfg_scale 0.5 --vfg_start 0.0 --vfg_end 1.0 --max_new_tokens 256
```
