open("/Volumes/Stadler100/2020-08-25/rpb1Snams2-em4-zsmv-02.czi");
run("Gaussian Blur...", "sigma=0.80 stack");
rename("1");
run("Merge Channels...", "c1=1 c2=1 create");
