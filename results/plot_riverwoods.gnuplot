set terminal pdf size 8cm,4cm font "Times, 11"
set output 'plot_riverwoods.pdf'

set xlabel 'Episode'
set ylabel 'Utility per episode'
set format x "%1.0fK"
set key bottom right
set style fill transparent solid 0.33 noborder

fn = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-forwardreturn-extranone-*)"
fb = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-forwardreturn-extraaccrued-*)"
bn = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-bothreturn-extranone-*)"
bb = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-bothreturn-extraaccrued-*)"
bbn = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-bothbuggyreturn-extranone-*)"
bbb = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-bothbuggyreturn-extraaccrued-*)"
tn = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-triplereturn-extranone-*)"
tb = "<(python3 avg_stats.py -1 out-fishwood-hidden50-lr0.001-triplereturn-extraaccrued-*)"

plot [0:20] [4:16] \
    fn using 1:3:4 with filledcu notitle lc rgb "#888888", fn using 1:2 with lines title 'u(R^+)' lc "#000000", \
    fb using 1:3:4 with filledcu notitle lc rgb "#888888", fb using 1:2 with lines title 'u(R^+), accrued' lc "#000000" lw 2, \
    bn using 1:3:4 with filledcu notitle lc rgb "#8888FF", bn using 1:2 with lines title 'u(R^- + R^+)' lc "#000088", \
    bb using 1:3:4 with filledcu notitle lc rgb "#8888FF", bb using 1:2 with lines title 'u(R^- + R^+), accrued' lc "#000088" lw 2, \
    bbn using 1:3:4 with filledcu notitle lc rgb "#88FF88", bbn using 1:2 with lines title 'u(R^-) + u(R^+)' lc "#008800", \
    bbb using 1:3:4 with filledcu notitle lc rgb "#88FF88", bbb using 1:2 with lines title 'u(R^-) + u(R^+), accrued' lc "#008800" lw 2, \
    tn using 1:3:4 with filledcu notitle lc rgb "#FF8888", tn using 1:2 with lines title 'u(R^- + R^+) - u(R^-)' lc "#880000", \
    tb using 1:3:4 with filledcu notitle lc rgb "#FF8888", tb using 1:2 with lines title 'u(R^- + R^+) - u(R^-), accrued' lc "#880000" lw 2
