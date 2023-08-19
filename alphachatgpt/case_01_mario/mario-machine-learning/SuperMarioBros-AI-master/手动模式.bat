set file="save\测试"
set inds=last

cmd /k "activate python37&&python -m retro.import ./rom&&python smb_ai.py --manual --load-file %file% --load-inds %inds%"