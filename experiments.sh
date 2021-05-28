#!/bin/bash
RUNS="1 2 3 4 5 6 7 8"
PYTHON="python3"

#ENV=random
#ENVN=RandomMOMDP-v0
#UTILITY="r1 + r2*r2 + r3*r3 - 0.1*r4"
ENV=fishwood
ENVN=FishWood-v0
UTILITY="min(r1, r2 // 2)"
LEN=100000

> commands_$ENV.sh

# Compare return computations
if true
then
    parallel --header : echo $PYTHON pg.py \
        --name "$ENV-hidden{hidden}-lr{lr}-{ret}return-extra{extra}-{run}" \
        --env $ENVN \
        --episodes $LEN \
        --avg 10 \
        --hidden {hidden} \
        --lr {lr} \
        --ret {ret} \
        --extra-state {extra} \
        --utility "\"\\\"$UTILITY\\\"\"" \
        "\">\"" /dev/null ">>" commands_$ENV.sh \
        ::: hidden 50 \
        ::: lr 0.001 \
        ::: ret forward both \
        ::: extra none timestep accrued both \
        ::: run $RUNS
fi

# Run all commands (Hydra version)
num_commands=$(wc -l commands_$ENV.sh | cut -d ' ' -f 1)
echo "qsub -t 1-$num_commands hydra_job.sh (with commands_$ENV.sh)"
