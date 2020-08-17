open("/Users/MStadler/Bioinformatics/Projects/Zelda/Quarantine_analysis/data/2020-07-30/em1-zsmv-07.czi");
run("Gaussian Blur...", "sigma=0.80 stack");
rename("1");
run("Merge Channels...", "c1=1 c2=1 create");
