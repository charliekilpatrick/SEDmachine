python testing/run_mcmc.py
python sedmachine.py --models villar_gw170817_3comp_mod  --instruments swift_uvot,sdss,ps1,ukirt,johnson,atlas,clear --clobber
python testing/plot_data.py
